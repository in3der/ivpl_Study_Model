# 0. 사용할 패키지 불러오기
import tensorflow as tf
import keras
from keras.datasets import cifar10
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model, Model
from keras import losses, layers
from keras.models import Sequential, Model
from keras.layers import (Conv2D, ZeroPadding2D, MaxPooling2D, GlobalAveragePooling2D,
                          MaxPool2D, Input, AveragePooling2D, Flatten,
                          Dense, Activation, Lambda, Dropout, BatchNormalization, concatenate)
from tensorflow.keras.preprocessing import image
from tensorflow.keras.callbacks import ModelCheckpoint
import random
import matplotlib.pyplot as plt
import numpy as np
from numpy import argmax
AUTOTUNE = tf.data.experimental.AUTOTUNE

# 1. 데이터 준비하기
# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

num_classes = 10
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# Split the data into training and validation sets using train_test_split
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

print(x_train.shape)
print(y_train.shape)
print(x_val.shape)
print(y_val.shape)
print(x_test.shape)
print(y_test.shape)

def preprocess_images(x, y):
    x = tf.image.resize(x, (224,224))
    x = x / 255.0
    return x, y

batch_size = 16

train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).map(preprocess_images).shuffle(buffer_size=10000).batch(batch_size).prefetch(AUTOTUNE)
val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val)).map(preprocess_images).batch(batch_size).prefetch(AUTOTUNE)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).map(preprocess_images).batch(batch_size).prefetch(AUTOTUNE)

train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)

print(train_ds)
print(val_ds)
print(test_ds)

# --------------------------------------------------------

# weight.h5 로드
model = Sequential()
model.add(layers.InputLayer(input_shape=(224, 224, 3)))
model.add(Conv2D(64,  (3, 3),  activation="relu", padding="same", ))
model.add(Conv2D(64,  (3, 3),  activation="relu", padding="same", ))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))
# Block 2
model.add(Conv2D(128, (3, 3), activation="relu", padding="same", ))
model.add(Conv2D(128, (3, 3), activation="relu", padding="same", ))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))
# Block 3)
model.add(Conv2D(256, (3, 3), activation="relu", padding="same", ))
model.add(Conv2D(256, (3, 3), activation="relu", padding="same", ))
model.add(Conv2D(256, (3, 3), activation="relu", padding="same", ))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))
# Block 4))
model.add(Conv2D(512, (3, 3), activation="relu", padding="same", ))
model.add(Conv2D(512, (3, 3), activation="relu", padding="same", ))
model.add(Conv2D(512, (3, 3), activation="relu", padding="same", ))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))
# Block 5))
model.add(Conv2D(512, (3, 3), activation="relu", padding="same", ))
model.add(Conv2D(512, (3, 3), activation="relu", padding="same", ))
model.add(Conv2D(512, (3, 3), activation="relu", padding="same", ))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))
# FC
model.add(Flatten())
model.add(Dense(4096, activation="relu", ))
model.add(Dense(4096, activation="relu", ))
model.add(Dense(1000, activation="softmax", ))

model.summary()



# --------------------------------------------------------

# 체크포인트 로드
# model.load_weights('/home/ivpl-d29/myProject/Study_Model/VGG16/ckpt/weights.h5')
# SGD = tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9, weight_decay=5e-4,)
# model.compile(optimizer=SGD, loss=losses.categorical_crossentropy, metrics=['accuracy'])

model.load_weights('/home/ivpl-d29/myProject/Study_Model/VGG16/ckpt/weights.h5')
sgd = tf.keras.optimizers.legacy.SGD(learning_rate=0.01, momentum=0.9, decay=0.0005)
model.compile(optimizer=sgd, loss=losses.categorical_crossentropy, metrics=['accuracy'])

# score = model.evaluate(test_ds, verbose=1)
#
# print('ckpt Test loss :', score[0])
# print('ckpt Test accuracy :', score[1])
# print(score[0:])

# 새로운 데이터 불러와 예측
new_image_path = '/home/ivpl-d29/dataset/cifar_test/tigershark.jpg'  # 새로운 이미지의 경로
img = image.load_img(new_image_path, target_size=(224,224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.0  # 이미지를 모델에 입력하기 전에 정규화

# 예측
predictions = model.predict(img_array)

def predict_class_percentage(predictions):
    def convert_to_class(prediction):
        return np.argmax(prediction)

    predicted_class = convert_to_class(predictions[0])
    print(f"Predicted Class: {predicted_class}")

    # 각 클래스에 대한 예측 확률 출력
    #class_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    class_names = ['Afghan_hound', 'African_chameleon', 'African_crocodile', 'African_elephant', 'African_grey', 'African_hunting_dog', 'Airedale', 'American_Staffordshire_terrier', 'American_alligator', 'American_black_bear', 'American_chameleon', 'American_coot', 'American_egret', 'American_lobster', 'Angora', 'Appenzeller', 'Arabian_camel', 'Arctic_fox', 'Australian_terrier', 'Band_Aid', 'Bedlington_terrier', 'Bernese_mountain_dog', 'Blenheim_spaniel', 'Border_collie', 'Border_terrier', 'Boston_bull', 'Bouvier_des_Flandres', 'Brabancon_griffon', 'Brittany_spaniel', 'CD_player', 'Cardigan', 'Chesapeake_Bay_retriever', 'Chihuahua', 'Christmas_stocking', 'Crock_Pot', 'Dandie_Dinmont', 'Doberman', 'Dungeness_crab', 'Dutch_oven', 'Egyptian_cat', 'English_foxhound', 'English_setter', 'English_springer', 'EntleBucher', 'Eskimo_dog', 'European_fire_salamander', 'European_gallinule', 'French_bulldog', 'French_horn', 'French_loaf', 'German_shepherd', 'German_short-haired_pointer', 'Gila_monster', 'Gordon_setter', 'Granny_Smith', 'Great_Dane', 'Great_Pyrenees', 'Greater_Swiss_Mountain_dog', 'Ibizan_hound', 'Indian_cobra', 'Indian_elephant', 'Irish_setter', 'Irish_terrier', 'Irish_water_spaniel', 'Irish_wolfhound', 'Italian_greyhound', 'Japanese_spaniel', 'Kerry_blue_terrier', 'Komodo_dragon', 'Labrador_retriever', 'Lakeland_terrier', 'Leonberg', 'Lhasa', 'Loafer', 'Madagascar_cat', 'Maltese_dog', 'Mexican_hairless', 'Model_T', 'Newfoundland', 'Norfolk_terrier', 'Norwegian_elkhound', 'Norwich_terrier', 'Old_English_sheepdog', 'Pekinese', 'Pembroke', 'Persian_cat', 'Petri_dish', 'Polaroid_camera', 'Pomeranian', 'Rhodesian_ridgeback', 'Rottweiler', 'Saint_Bernard', 'Saluki', 'Samoyed', 'Scotch_terrier', 'Scottish_deerhound', 'Sealyham_terrier', 'Shetland_sheepdog', 'Shih-Tzu', 'Siamese_cat', 'Siberian_husky', 'Staffordshire_bullterrier', 'Sussex_spaniel', 'Tibetan_mastiff', 'Tibetan_terrier', 'Walker_hound', 'Weimaraner', 'Welsh_springer_spaniel', 'West_Highland_white_terrier', 'Windsor_tie', 'Yorkshire_terrier', 'abacus', 'abaya', 'academic_gown', 'accordion', 'acorn', 'acorn_squash', 'acoustic_guitar', 'admiral', 'affenpinscher', 'agama', 'agaric', 'aircraft_carrier', 'airliner', 'airship', 'albatross', 'alligator_lizard', 'alp', 'altar', 'ambulance', 'amphibian', 'analog_clock', 'anemone_fish', 'ant', 'apiary', 'apron', 'armadillo', 'artichoke', 'ashcan', 'assault_rifle', 'axolotl', 'baboon', 'backpack', 'badger', 'bagel', 'bakery', 'balance_beam', 'bald_eagle', 'balloon', 'ballplayer', 'ballpoint', 'banana', 'banded_gecko', 'banjo', 'bannister', 'barbell', 'barber_chair', 'barbershop', 'barn', 'barn_spider', 'barometer', 'barracouta', 'barrel', 'barrow', 'baseball', 'basenji', 'basketball', 'basset', 'bassinet', 'bassoon', 'bath_towel', 'bathing_cap', 'bathtub', 'beach_wagon', 'beacon', 'beagle', 'beaker', 'bearskin', 'beaver', 'bee', 'bee_eater', 'beer_bottle', 'beer_glass', 'bell_cote', 'bell_pepper', 'bib', 'bicycle-built-for-two', 'bighorn', 'bikini', 'binder', 'binoculars', 'birdhouse', 'bison', 'bittern', 'black-and-tan_coonhound', 'black-footed_ferret', 'black_and_gold_garden_spider', 'black_grouse', 'black_stork', 'black_swan', 'black_widow', 'bloodhound', 'bluetick', 'boa_constrictor', 'boathouse', 'bobsled', 'bolete', 'bolo_tie', 'bonnet', 'book_jacket', 'bookcase', 'bookshop', 'borzoi', 'bottlecap', 'bow', 'bow_tie', 'box_turtle', 'boxer', 'brain_coral', 'brambling', 'brass', 'brassiere', 'breakwater', 'breastplate', 'briard', 'broccoli', 'broom', 'brown_bear', 'bubble', 'bucket', 'buckeye', 'buckle', 'bulbul', 'bull_mastiff', 'bullet_train', 'bulletproof_vest', 'bullfrog', 'burrito', 'bustard', 'butcher_shop', 'butternut_squash', 'cab', 'cabbage_butterfly', 'cairn', 'caldron', 'can_opener', 'candle', 'cannon', 'canoe', 'capuchin', 'car_mirror', 'car_wheel', 'carbonara', 'cardigan', 'cardoon', 'carousel', "carpenter's_kit", 'carton', 'cash_machine', 'cassette', 'cassette_player', 'castle', 'catamaran', 'cauliflower', 'cello', 'cellular_telephone', 'centipede', 'chain', 'chain_mail', 'chain_saw', 'chainlink_fence', 'chambered_nautilus', 'cheeseburger', 'cheetah', 'chest', 'chickadee', 'chiffonier', 'chime', 'chimpanzee', 'china_cabinet', 'chiton', 'chocolate_sauce', 'chow', 'church', 'cicada', 'cinema', 'cleaver', 'cliff', 'cliff_dwelling', 'cloak', 'clog', 'clumber', 'cock', 'cocker_spaniel', 'cockroach', 'cocktail_shaker', 'coffee_mug', 'coffeepot', 'coho', 'coil', 'collie', 'colobus', 'combination_lock', 'comic_book', 'common_iguana', 'common_newt', 'computer_keyboard', 'conch', 'confectionery', 'consomme', 'container_ship', 'convertible', 'coral_fungus', 'coral_reef', 'corkscrew', 'corn', 'cornet', 'coucal', 'cougar', 'cowboy_boot', 'cowboy_hat', 'coyote', 'cradle', 'crane', 'crane_bird', 'crash_helmet', 'crate', 'crayfish', 'crib', 'cricket', 'croquet_ball', 'crossword_puzzle', 'crutch', 'cucumber', 'cuirass', 'cup', 'curly-coated_retriever', 'custard_apple', 'daisy', 'dalmatian', 'dam', 'damselfly', 'desk', 'desktop_computer', 'dhole', 'dial_telephone', 'diamondback', 'diaper', 'digital_clock', 'digital_watch', 'dingo', 'dining_table', 'dishrag', 'dishwasher', 'disk_brake', 'dock', 'dogsled', 'dome', 'doormat', 'dough', 'dowitcher', 'dragonfly', 'drake', 'drilling_platform', 'drum', 'drumstick', 'dugong', 'dumbbell', 'dung_beetle', 'ear', 'earthstar', 'echidna', 'eel', 'eft', 'eggnog', 'electric_fan', 'electric_guitar', 'electric_locomotive', 'electric_ray', 'entertainment_center', 'envelope', 'espresso', 'espresso_maker', 'face_powder', 'feather_boa', 'fiddler_crab', 'fig', 'file', 'fire_engine', 'fire_screen', 'fireboat', 'flagpole', 'flamingo', 'flat-coated_retriever', 'flatworm', 'flute', 'fly', 'folding_chair', 'football_helmet', 'forklift', 'fountain', 'fountain_pen', 'four-poster', 'fox_squirrel', 'freight_car', 'frilled_lizard', 'frying_pan', 'fur_coat', 'gar', 'garbage_truck', 'garden_spider', 'garter_snake', 'gas_pump', 'gasmask', 'gazelle', 'geyser', 'giant_panda', 'giant_schnauzer', 'gibbon', 'go-kart', 'goblet', 'golden_retriever', 'goldfinch', 'goldfish', 'golf_ball', 'golfcart', 'gondola', 'gong', 'goose', 'gorilla', 'gown', 'grand_piano', 'grasshopper', 'great_grey_owl', 'great_white_shark', 'green_lizard', 'green_mamba', 'green_snake', 'greenhouse', 'grey_fox', 'grey_whale', 'grille', 'grocery_store', 'groenendael', 'groom', 'ground_beetle', 'guacamole', 'guenon', 'guillotine', 'guinea_pig', 'gyromitra', 'hair_slide', 'hair_spray', 'half_track', 'hammer', 'hammerhead', 'hamper', 'hamster', 'hand-held_computer', 'hand_blower', 'handkerchief', 'hard_disc', 'hare', 'harmonica', 'harp', 'hartebeest', 'harvester', 'harvestman', 'hatchet', 'hay', 'head_cabbage', 'hen', 'hen-of-the-woods', 'hermit_crab', 'hip', 'hippopotamus', 'hog', 'hognose_snake', 'holster', 'home_theater', 'honeycomb', 'hook', 'hoopskirt', 'horizontal_bar', 'hornbill', 'horned_viper', 'horse_cart', 'hot_pot', 'hotdog', 'hourglass', 'house_finch', 'howler_monkey', 'hummingbird', 'hyena', 'iPod', 'ibex', 'ice_bear', 'ice_cream', 'ice_lolly', 'impala', 'indigo_bunting', 'indri', 'iron', 'isopod', 'jacamar', "jack-o'-lantern", 'jackfruit', 'jaguar', 'jay', 'jean', 'jeep', 'jellyfish', 'jersey', 'jigsaw_puzzle', 'jinrikisha', 'joystick', 'junco', 'keeshond', 'kelpie', 'killer_whale', 'kimono', 'king_crab', 'king_penguin', 'king_snake', 'kit_fox', 'kite', 'knee_pad', 'knot', 'koala', 'komondor', 'kuvasz', 'lab_coat', 'lacewing', 'ladle', 'ladybug', 'lakeside', 'lampshade', 'langur', 'laptop', 'lawn_mower', 'leaf_beetle', 'leafhopper', 'leatherback_turtle', 'lemon', 'lens_cap', 'leopard', 'lesser_panda', 'letter_opener', 'library', 'lifeboat', 'lighter', 'limousine', 'limpkin', 'liner', 'lion', 'lionfish', 'lipstick', 'little_blue_heron', 'llama', 'loggerhead', 'long-horned_beetle', 'lorikeet', 'lotion', 'loudspeaker', 'loupe', 'lumbermill', 'lycaenid', 'lynx', 'macaque', 'macaw', 'magnetic_compass', 'magpie', 'mailbag', 'mailbox', 'maillot_1', 'maillot_2', 'malamute', 'malinois', 'manhole_cover', 'mantis', 'maraca', 'marimba', 'marmoset', 'marmot', 'mashed_potato', 'mask', 'matchstick', 'maypole', 'maze', 'measuring_cup', 'meat_loaf', 'medicine_chest', 'meerkat', 'megalith', 'menu', 'microphone', 'microwave', 'military_uniform', 'milk_can', 'miniature_pinscher', 'miniature_poodle', 'miniature_schnauzer', 'minibus', 'miniskirt', 'minivan', 'mink', 'missile', 'mitten', 'mixing_bowl', 'mobile_home', 'modem', 'monarch', 'monastery', 'mongoose', 'monitor', 'moped', 'mortar', 'mortarboard', 'mosque', 'mosquito_net', 'motor_scooter', 'mountain_bike', 'mountain_tent', 'mouse', 'mousetrap', 'moving_van', 'mud_turtle', 'mushroom', 'muzzle', 'nail', 'neck_brace', 'necklace', 'nematode', 'night_snake', 'nipple', 'notebook', 'obelisk', 'oboe', 'ocarina', 'odometer', 'oil_filter', 'orange', 'orangutan', 'organ', 'oscilloscope', 'ostrich', 'otter', 'otterhound', 'overskirt', 'ox', 'oxcart', 'oxygen_mask', 'oystercatcher', 'packet', 'paddle', 'paddlewheel', 'padlock', 'paintbrush', 'pajama', 'palace', 'panpipe', 'paper_towel', 'papillon', 'parachute', 'parallel_bars', 'park_bench', 'parking_meter', 'partridge', 'passenger_car', 'patas', 'patio', 'pay-phone', 'peacock', 'pedestal', 'pelican', 'pencil_box', 'pencil_sharpener', 'perfume', 'photocopier', 'pick', 'pickelhaube', 'picket_fence', 'pickup', 'pier', 'piggy_bank', 'pill_bottle', 'pillow', 'pineapple', 'ping-pong_ball', 'pinwheel', 'pirate', 'pitcher', 'pizza', 'plane', 'planetarium', 'plastic_bag', 'plate', 'plate_rack', 'platypus', 'plow', 'plunger', 'pole', 'polecat', 'police_van', 'pomegranate', 'poncho', 'pool_table', 'pop_bottle', 'porcupine', 'pot', 'potpie', "potter's_wheel", 'power_drill', 'prairie_chicken', 'prayer_rug', 'pretzel', 'printer', 'prison', 'proboscis_monkey', 'projectile', 'projector', 'promontory', 'ptarmigan', 'puck', 'puffer', 'pug', 'punching_bag', 'purse', 'quail', 'quill', 'quilt', 'racer', 'racket', 'radiator', 'radio', 'radio_telescope', 'rain_barrel', 'ram', 'rapeseed', 'recreational_vehicle', 'red-backed_sandpiper', 'red-breasted_merganser', 'red_fox', 'red_wine', 'red_wolf', 'redbone', 'redshank', 'reel', 'reflex_camera', 'refrigerator', 'remote_control', 'restaurant', 'revolver', 'rhinoceros_beetle', 'rifle', 'ringlet', 'ringneck_snake', 'robin', 'rock_beauty', 'rock_crab', 'rock_python', 'rocking_chair', 'rotisserie', 'rubber_eraser', 'ruddy_turnstone', 'ruffed_grouse', 'rugby_ball', 'rule', 'running_shoe', 'safe', 'safety_pin', 'saltshaker', 'sandal', 'sandbar', 'sarong', 'sax', 'scabbard', 'scale', 'schipperke', 'school_bus', 'schooner', 'scoreboard', 'scorpion', 'screen', 'screw', 'screwdriver', 'scuba_diver', 'sea_anemone', 'sea_cucumber', 'sea_lion', 'sea_slug', 'sea_snake', 'sea_urchin', 'seashore', 'seat_belt', 'sewing_machine', 'shield', 'shoe_shop', 'shoji', 'shopping_basket', 'shopping_cart', 'shovel', 'shower_cap', 'shower_curtain', 'siamang', 'sidewinder', 'silky_terrier', 'ski', 'ski_mask', 'skunk', 'sleeping_bag', 'slide_rule', 'sliding_door', 'slot', 'sloth_bear', 'slug', 'snail', 'snorkel', 'snow_leopard', 'snowmobile', 'snowplow', 'soap_dispenser', 'soccer_ball', 'sock', 'soft-coated_wheaten_terrier', 'solar_dish', 'sombrero', 'sorrel', 'soup_bowl', 'space_bar', 'space_heater', 'space_shuttle', 'spaghetti_squash', 'spatula', 'speedboat', 'spider_monkey', 'spider_web', 'spindle', 'spiny_lobster', 'spoonbill', 'sports_car', 'spotlight', 'spotted_salamander', 'squirrel_monkey', 'stage', 'standard_poodle', 'standard_schnauzer', 'starfish', 'steam_locomotive', 'steel_arch_bridge', 'steel_drum', 'stethoscope', 'stingray', 'stinkhorn', 'stole', 'stone_wall', 'stopwatch', 'stove', 'strainer', 'strawberry', 'street_sign', 'streetcar', 'stretcher', 'studio_couch', 'stupa', 'sturgeon', 'submarine', 'suit', 'sulphur-crested_cockatoo', 'sulphur_butterfly', 'sundial', 'sunglass', 'sunglasses', 'sunscreen', 'suspension_bridge', 'swab', 'sweatshirt', 'swimming_trunks', 'swing', 'switch', 'syringe', 'tabby', 'table_lamp', 'tailed_frog', 'tank', 'tape_player', 'tarantula', 'teapot', 'teddy', 'television', 'tench', 'tennis_ball', 'terrapin', 'thatch', 'theater_curtain', 'thimble', 'three-toed_sloth', 'thresher', 'throne', 'thunder_snake', 'tick', 'tiger', 'tiger_beetle', 'tiger_cat', 'tiger_shark', 'tile_roof', 'timber_wolf', 'titi', 'toaster', 'tobacco_shop', 'toilet_seat', 'toilet_tissue', 'torch', 'totem_pole', 'toucan', 'tow_truck', 'toy_poodle', 'toy_terrier', 'toyshop', 'tractor', 'traffic_light', 'trailer_truck', 'tray', 'tree_frog', 'trench_coat', 'triceratops', 'tricycle', 'trifle', 'trilobite', 'trimaran', 'tripod', 'triumphal_arch', 'trolleybus', 'trombone', 'tub', 'turnstile', 'tusker', 'typewriter_keyboard', 'umbrella', 'unicycle', 'upright', 'vacuum', 'valley', 'vase', 'vault', 'velvet', 'vending_machine', 'vestment', 'viaduct', 'vine_snake', 'violin', 'vizsla', 'volcano', 'volleyball', 'vulture', 'waffle_iron', 'walking_stick', 'wall_clock', 'wallaby', 'wallet', 'wardrobe', 'warplane', 'warthog', 'washbasin', 'washer', 'water_bottle', 'water_buffalo', 'water_jug', 'water_ouzel', 'water_snake', 'water_tower', 'weasel', 'web_site', 'weevil', 'whippet', 'whiptail', 'whiskey_jug', 'whistle', 'white_stork', 'white_wolf', 'wig', 'wild_boar', 'window_screen', 'window_shade', 'wine_bottle', 'wing', 'wire-haired_fox_terrier', 'wok', 'wolf_spider', 'wombat', 'wood_rabbit', 'wooden_spoon', 'wool', 'worm_fence', 'wreck', 'yawl', "yellow_lady's_slipper", 'yurt', 'zebra', 'zucchini']

    # for i in range(len(class_names)):
    #     print(f"{class_names[i]}: {predictions[0][i] * 100:.2f}%")


    # 클래스명과 예측 확률을 묶은 튜플 리스트 생성
    class_prob_tuples = [(class_name, prob) for class_name, prob in zip(class_names, predictions[0])]

    # 예측 확률을 기준으로 정렬
    sorted_class_prob_tuples = sorted(class_prob_tuples, key=lambda x: x[1], reverse=True)

    # 정렬된 결과 출력
    for class_name, prob in sorted_class_prob_tuples:
        print(f"{class_name}: {prob * 100:.2f}%")
    # 이미지 시각화
    plt.imshow(img)
    plt.title(f"Predicted Class: {predicted_class}")
    plt.show()
# 예측 결과 출력
predict_class_percentage(predictions)
