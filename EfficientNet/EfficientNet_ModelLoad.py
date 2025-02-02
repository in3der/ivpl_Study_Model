import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.nn import AvgPool2d, MaxPool2d, ReLU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'device : {device}')
from torchsummary import summary
from tqdm import tqdm
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import os
from torchvision.ops import StochasticDepth

# GPU가 사용 가능한지 확인
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# 데이터 전처리 및 augmentation
data_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

# 데이터셋 로드
data_dir = '/home/ivpl-d29/dataset/imagenet'
train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=data_transforms)
val_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=data_transforms)
test_dataset = datasets.ImageFolder(os.path.join(data_dir, 'test'), transform=data_transforms)

# 데이터 로더
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4, drop_last=True, pin_memory=True, prefetch_factor=2)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=True, num_workers=4, drop_last=True, pin_memory=True, prefetch_factor=2)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=True, num_workers=4, drop_last=True, pin_memory=True, prefetch_factor=2)

# cifar10
transform = transforms.Compose([
    transforms.Resize((40, 40)),
    transforms.RandomCrop((32, 32)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)



# 모든 클래스 1000개 이름
class_names = ['Afghan_hound', 'African_chameleon', 'African_crocodile', 'African_elephant', 'African_grey',
               'African_hunting_dog', 'Airedale', 'American_Staffordshire_terrier', 'American_alligator',
               'American_black_bear', 'American_chameleon', 'American_coot', 'American_egret', 'American_lobster',
               'Angora', 'Appenzeller', 'Arabian_camel', 'Arctic_fox', 'Australian_terrier', 'Band_Aid',
               'Bedlington_terrier', 'Bernese_mountain_dog', 'Blenheim_spaniel', 'Border_collie', 'Border_terrier',
               'Boston_bull', 'Bouvier_des_Flandres', 'Brabancon_griffon', 'Brittany_spaniel', 'CD_player', 'Cardigan',
               'Chesapeake_Bay_retriever', 'Chihuahua', 'Christmas_stocking', 'Crock_Pot', 'Dandie_Dinmont', 'Doberman',
               'Dungeness_crab', 'Dutch_oven', 'Egyptian_cat', 'English_foxhound', 'English_setter', 'English_springer',
               'EntleBucher', 'Eskimo_dog', 'European_fire_salamander', 'European_gallinule', 'French_bulldog',
               'French_horn', 'French_loaf', 'German_shepherd', 'German_short-haired_pointer', 'Gila_monster',
               'Gordon_setter', 'Granny_Smith', 'Great_Dane', 'Great_Pyrenees', 'Greater_Swiss_Mountain_dog',
               'Ibizan_hound', 'Indian_cobra', 'Indian_elephant', 'Irish_setter', 'Irish_terrier',
               'Irish_water_spaniel', 'Irish_wolfhound', 'Italian_greyhound', 'Japanese_spaniel', 'Kerry_blue_terrier',
               'Komodo_dragon', 'Labrador_retriever', 'Lakeland_terrier', 'Leonberg', 'Lhasa', 'Loafer',
               'Madagascar_cat', 'Maltese_dog', 'Mexican_hairless', 'Model_T', 'Newfoundland', 'Norfolk_terrier',
               'Norwegian_elkhound', 'Norwich_terrier', 'Old_English_sheepdog', 'Pekinese', 'Pembroke', 'Persian_cat',
               'Petri_dish', 'Polaroid_camera', 'Pomeranian', 'Rhodesian_ridgeback', 'Rottweiler', 'Saint_Bernard',
               'Saluki', 'Samoyed', 'Scotch_terrier', 'Scottish_deerhound', 'Sealyham_terrier', 'Shetland_sheepdog',
               'Shih-Tzu', 'Siamese_cat', 'Siberian_husky', 'Staffordshire_bullterrier', 'Sussex_spaniel',
               'Tibetan_mastiff', 'Tibetan_terrier', 'Walker_hound', 'Weimaraner', 'Welsh_springer_spaniel',
               'West_Highland_white_terrier', 'Windsor_tie', 'Yorkshire_terrier', 'abacus', 'abaya', 'academic_gown',
               'accordion', 'acorn', 'acorn_squash', 'acoustic_guitar', 'admiral', 'affenpinscher', 'agama', 'agaric',
               'aircraft_carrier', 'airliner', 'airship', 'albatross', 'alligator_lizard', 'alp', 'altar', 'ambulance',
               'amphibian', 'analog_clock', 'anemone_fish', 'ant', 'apiary', 'apron', 'armadillo', 'artichoke',
               'ashcan', 'assault_rifle', 'axolotl', 'baboon', 'backpack', 'badger', 'bagel', 'bakery', 'balance_beam',
               'bald_eagle', 'balloon', 'ballplayer', 'ballpoint', 'banana', 'banded_gecko', 'banjo', 'bannister',
               'barbell', 'barber_chair', 'barbershop', 'barn', 'barn_spider', 'barometer', 'barracouta', 'barrel',
               'barrow', 'baseball', 'basenji', 'basketball', 'basset', 'bassinet', 'bassoon', 'bath_towel',
               'bathing_cap', 'bathtub', 'beach_wagon', 'beacon', 'beagle', 'beaker', 'bearskin', 'beaver', 'bee',
               'bee_eater', 'beer_bottle', 'beer_glass', 'bell_cote', 'bell_pepper', 'bib', 'bicycle-built-for-two',
               'bighorn', 'bikini', 'binder', 'binoculars', 'birdhouse', 'bison', 'bittern', 'black-and-tan_coonhound',
               'black-footed_ferret', 'black_and_gold_garden_spider', 'black_grouse', 'black_stork', 'black_swan',
               'black_widow', 'bloodhound', 'bluetick', 'boa_constrictor', 'boathouse', 'bobsled', 'bolete', 'bolo_tie',
               'bonnet', 'book_jacket', 'bookcase', 'bookshop', 'borzoi', 'bottlecap', 'bow', 'bow_tie', 'box_turtle',
               'boxer', 'brain_coral', 'brambling', 'brass', 'brassiere', 'breakwater', 'breastplate', 'briard',
               'broccoli', 'broom', 'brown_bear', 'bubble', 'bucket', 'buckeye', 'buckle', 'bulbul', 'bull_mastiff',
               'bullet_train', 'bulletproof_vest', 'bullfrog', 'burrito', 'bustard', 'butcher_shop', 'butternut_squash',
               'cab', 'cabbage_butterfly', 'cairn', 'caldron', 'can_opener', 'candle', 'cannon', 'canoe', 'capuchin',
               'car_mirror', 'car_wheel', 'carbonara', 'cardigan', 'cardoon', 'carousel', "carpenter's_kit", 'carton',
               'cash_machine', 'cassette', 'cassette_player', 'castle', 'catamaran', 'cauliflower', 'cello',
               'cellular_telephone', 'centipede', 'chain', 'chain_mail', 'chain_saw', 'chainlink_fence',
               'chambered_nautilus', 'cheeseburger', 'cheetah', 'chest', 'chickadee', 'chiffonier', 'chime',
               'chimpanzee', 'china_cabinet', 'chiton', 'chocolate_sauce', 'chow', 'church', 'cicada', 'cinema',
               'cleaver', 'cliff', 'cliff_dwelling', 'cloak', 'clog', 'clumber', 'cock', 'cocker_spaniel', 'cockroach',
               'cocktail_shaker', 'coffee_mug', 'coffeepot', 'coho', 'coil', 'collie', 'colobus', 'combination_lock',
               'comic_book', 'common_iguana', 'common_newt', 'computer_keyboard', 'conch', 'confectionery', 'consomme',
               'container_ship', 'convertible', 'coral_fungus', 'coral_reef', 'corkscrew', 'corn', 'cornet', 'coucal',
               'cougar', 'cowboy_boot', 'cowboy_hat', 'coyote', 'cradle', 'crane', 'crane_bird', 'crash_helmet',
               'crate', 'crayfish', 'crib', 'cricket', 'croquet_ball', 'crossword_puzzle', 'crutch', 'cucumber',
               'cuirass', 'cup', 'curly-coated_retriever', 'custard_apple', 'daisy', 'dalmatian', 'dam', 'damselfly',
               'desk', 'desktop_computer', 'dhole', 'dial_telephone', 'diamondback', 'diaper', 'digital_clock',
               'digital_watch', 'dingo', 'dining_table', 'dishrag', 'dishwasher', 'disk_brake', 'dock', 'dogsled',
               'dome', 'doormat', 'dough', 'dowitcher', 'dragonfly', 'drake', 'drilling_platform', 'drum', 'drumstick',
               'dugong', 'dumbbell', 'dung_beetle', 'ear', 'earthstar', 'echidna', 'eel', 'eft', 'eggnog',
               'electric_fan', 'electric_guitar', 'electric_locomotive', 'electric_ray', 'entertainment_center',
               'envelope', 'espresso', 'espresso_maker', 'face_powder', 'feather_boa', 'fiddler_crab', 'fig', 'file',
               'fire_engine', 'fire_screen', 'fireboat', 'flagpole', 'flamingo', 'flat-coated_retriever', 'flatworm',
               'flute', 'fly', 'folding_chair', 'football_helmet', 'forklift', 'fountain', 'fountain_pen',
               'four-poster', 'fox_squirrel', 'freight_car', 'frilled_lizard', 'frying_pan', 'fur_coat', 'gar',
               'garbage_truck', 'garden_spider', 'garter_snake', 'gas_pump', 'gasmask', 'gazelle', 'geyser',
               'giant_panda', 'giant_schnauzer', 'gibbon', 'go-kart', 'goblet', 'golden_retriever', 'goldfinch',
               'goldfish', 'golf_ball', 'golfcart', 'gondola', 'gong', 'goose', 'gorilla', 'gown', 'grand_piano',
               'grasshopper', 'great_grey_owl', 'great_white_shark', 'green_lizard', 'green_mamba', 'green_snake',
               'greenhouse', 'grey_fox', 'grey_whale', 'grille', 'grocery_store', 'groenendael', 'groom',
               'ground_beetle', 'guacamole', 'guenon', 'guillotine', 'guinea_pig', 'gyromitra', 'hair_slide',
               'hair_spray', 'half_track', 'hammer', 'hammerhead', 'hamper', 'hamster', 'hand-held_computer',
               'hand_blower', 'handkerchief', 'hard_disc', 'hare', 'harmonica', 'harp', 'hartebeest', 'harvester',
               'harvestman', 'hatchet', 'hay', 'head_cabbage', 'hen', 'hen-of-the-woods', 'hermit_crab', 'hip',
               'hippopotamus', 'hog', 'hognose_snake', 'holster', 'home_theater', 'honeycomb', 'hook', 'hoopskirt',
               'horizontal_bar', 'hornbill', 'horned_viper', 'horse_cart', 'hot_pot', 'hotdog', 'hourglass',
               'house_finch', 'howler_monkey', 'hummingbird', 'hyena', 'iPod', 'ibex', 'ice_bear', 'ice_cream',
               'ice_lolly', 'impala', 'indigo_bunting', 'indri', 'iron', 'isopod', 'jacamar', "jack-o'-lantern",
               'jackfruit', 'jaguar', 'jay', 'jean', 'jeep', 'jellyfish', 'jersey', 'jigsaw_puzzle', 'jinrikisha',
               'joystick', 'junco', 'keeshond', 'kelpie', 'killer_whale', 'kimono', 'king_crab', 'king_penguin',
               'king_snake', 'kit_fox', 'kite', 'knee_pad', 'knot', 'koala', 'komondor', 'kuvasz', 'lab_coat',
               'lacewing', 'ladle', 'ladybug', 'lakeside', 'lampshade', 'langur', 'laptop', 'lawn_mower', 'leaf_beetle',
               'leafhopper', 'leatherback_turtle', 'lemon', 'lens_cap', 'leopard', 'lesser_panda', 'letter_opener',
               'library', 'lifeboat', 'lighter', 'limousine', 'limpkin', 'liner', 'lion', 'lionfish', 'lipstick',
               'little_blue_heron', 'llama', 'loggerhead', 'long-horned_beetle', 'lorikeet', 'lotion', 'loudspeaker',
               'loupe', 'lumbermill', 'lycaenid', 'lynx', 'macaque', 'macaw', 'magnetic_compass', 'magpie', 'mailbag',
               'mailbox', 'maillot_1', 'maillot_2', 'malamute', 'malinois', 'manhole_cover', 'mantis', 'maraca',
               'marimba', 'marmoset', 'marmot', 'mashed_potato', 'mask', 'matchstick', 'maypole', 'maze',
               'measuring_cup', 'meat_loaf', 'medicine_chest', 'meerkat', 'megalith', 'menu', 'microphone', 'microwave',
               'military_uniform', 'milk_can', 'miniature_pinscher', 'miniature_poodle', 'miniature_schnauzer',
               'minibus', 'miniskirt', 'minivan', 'mink', 'missile', 'mitten', 'mixing_bowl', 'mobile_home', 'modem',
               'monarch', 'monastery', 'mongoose', 'monitor', 'moped', 'mortar', 'mortarboard', 'mosque',
               'mosquito_net', 'motor_scooter', 'mountain_bike', 'mountain_tent', 'mouse', 'mousetrap', 'moving_van',
               'mud_turtle', 'mushroom', 'muzzle', 'nail', 'neck_brace', 'necklace', 'nematode', 'night_snake',
               'nipple', 'notebook', 'obelisk', 'oboe', 'ocarina', 'odometer', 'oil_filter', 'orange', 'orangutan',
               'organ', 'oscilloscope', 'ostrich', 'otter', 'otterhound', 'overskirt', 'ox', 'oxcart', 'oxygen_mask',
               'oystercatcher', 'packet', 'paddle', 'paddlewheel', 'padlock', 'paintbrush', 'pajama', 'palace',
               'panpipe', 'paper_towel', 'papillon', 'parachute', 'parallel_bars', 'park_bench', 'parking_meter',
               'partridge', 'passenger_car', 'patas', 'patio', 'pay-phone', 'peacock', 'pedestal', 'pelican',
               'pencil_box', 'pencil_sharpener', 'perfume', 'photocopier', 'pick', 'pickelhaube', 'picket_fence',
               'pickup', 'pier', 'piggy_bank', 'pill_bottle', 'pillow', 'pineapple', 'ping-pong_ball', 'pinwheel',
               'pirate', 'pitcher', 'pizza', 'plane', 'planetarium', 'plastic_bag', 'plate', 'plate_rack', 'platypus',
               'plow', 'plunger', 'pole', 'polecat', 'police_van', 'pomegranate', 'poncho', 'pool_table', 'pop_bottle',
               'porcupine', 'pot', 'potpie', "potter's_wheel", 'power_drill', 'prairie_chicken', 'prayer_rug',
               'pretzel', 'printer', 'prison', 'proboscis_monkey', 'projectile', 'projector', 'promontory', 'ptarmigan',
               'puck', 'puffer', 'pug', 'punching_bag', 'purse', 'quail', 'quill', 'quilt', 'racer', 'racket',
               'radiator', 'radio', 'radio_telescope', 'rain_barrel', 'ram', 'rapeseed', 'recreational_vehicle',
               'red-backed_sandpiper', 'red-breasted_merganser', 'red_fox', 'red_wine', 'red_wolf', 'redbone',
               'redshank', 'reel', 'reflex_camera', 'refrigerator', 'remote_control', 'restaurant', 'revolver',
               'rhinoceros_beetle', 'rifle', 'ringlet', 'ringneck_snake', 'robin', 'rock_beauty', 'rock_crab',
               'rock_python', 'rocking_chair', 'rotisserie', 'rubber_eraser', 'ruddy_turnstone', 'ruffed_grouse',
               'rugby_ball', 'rule', 'running_shoe', 'safe', 'safety_pin', 'saltshaker', 'sandal', 'sandbar', 'sarong',
               'sax', 'scabbard', 'scale', 'schipperke', 'school_bus', 'schooner', 'scoreboard', 'scorpion', 'screen',
               'screw', 'screwdriver', 'scuba_diver', 'sea_anemone', 'sea_cucumber', 'sea_lion', 'sea_slug',
               'sea_snake', 'sea_urchin', 'seashore', 'seat_belt', 'sewing_machine', 'shield', 'shoe_shop', 'shoji',
               'shopping_basket', 'shopping_cart', 'shovel', 'shower_cap', 'shower_curtain', 'siamang', 'sidewinder',
               'silky_terrier', 'ski', 'ski_mask', 'skunk', 'sleeping_bag', 'slide_rule', 'sliding_door', 'slot',
               'sloth_bear', 'slug', 'snail', 'snorkel', 'snow_leopard', 'snowmobile', 'snowplow', 'soap_dispenser',
               'soccer_ball', 'sock', 'soft-coated_wheaten_terrier', 'solar_dish', 'sombrero', 'sorrel', 'soup_bowl',
               'space_bar', 'space_heater', 'space_shuttle', 'spaghetti_squash', 'spatula', 'speedboat',
               'spider_monkey', 'spider_web', 'spindle', 'spiny_lobster', 'spoonbill', 'sports_car', 'spotlight',
               'spotted_salamander', 'squirrel_monkey', 'stage', 'standard_poodle', 'standard_schnauzer', 'starfish',
               'steam_locomotive', 'steel_arch_bridge', 'steel_drum', 'stethoscope', 'stingray', 'stinkhorn', 'stole',
               'stone_wall', 'stopwatch', 'stove', 'strainer', 'strawberry', 'street_sign', 'streetcar', 'stretcher',
               'studio_couch', 'stupa', 'sturgeon', 'submarine', 'suit', 'sulphur-crested_cockatoo',
               'sulphur_butterfly', 'sundial', 'sunglass', 'sunglasses', 'sunscreen', 'suspension_bridge', 'swab',
               'sweatshirt', 'swimming_trunks', 'swing', 'switch', 'syringe', 'tabby', 'table_lamp', 'tailed_frog',
               'tank', 'tape_player', 'tarantula', 'teapot', 'teddy', 'television', 'tench', 'tennis_ball', 'terrapin',
               'thatch', 'theater_curtain', 'thimble', 'three-toed_sloth', 'thresher', 'throne', 'thunder_snake',
               'tick', 'tiger', 'tiger_beetle', 'tiger_cat', 'tiger_shark', 'tile_roof', 'timber_wolf', 'titi',
               'toaster', 'tobacco_shop', 'toilet_seat', 'toilet_tissue', 'torch', 'totem_pole', 'toucan', 'tow_truck',
               'toy_poodle', 'toy_terrier', 'toyshop', 'tractor', 'traffic_light', 'trailer_truck', 'tray', 'tree_frog',
               'trench_coat', 'triceratops', 'tricycle', 'trifle', 'trilobite', 'trimaran', 'tripod', 'triumphal_arch',
               'trolleybus', 'trombone', 'tub', 'turnstile', 'tusker', 'typewriter_keyboard', 'umbrella', 'unicycle',
               'upright', 'vacuum', 'valley', 'vase', 'vault', 'velvet', 'vending_machine', 'vestment', 'viaduct',
               'vine_snake', 'violin', 'vizsla', 'volcano', 'volleyball', 'vulture', 'waffle_iron', 'walking_stick',
               'wall_clock', 'wallaby', 'wallet', 'wardrobe', 'warplane', 'warthog', 'washbasin', 'washer',
               'water_bottle', 'water_buffalo', 'water_jug', 'water_ouzel', 'water_snake', 'water_tower', 'weasel',
               'web_site', 'weevil', 'whippet', 'whiptail', 'whiskey_jug', 'whistle', 'white_stork', 'white_wolf',
               'wig', 'wild_boar', 'window_screen', 'window_shade', 'wine_bottle', 'wing', 'wire-haired_fox_terrier',
               'wok', 'wolf_spider', 'wombat', 'wood_rabbit', 'wooden_spoon', 'wool', 'worm_fence', 'wreck', 'yawl',
               "yellow_lady's_slipper", 'yurt', 'zebra', 'zucchini']
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


class SiLU(nn.Module):  # ReLU 대신 사용하는 activation 함수
    def __init__(self):
        super(SiLU, self).__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return x * self.sigmoid(x)


class SEBlock(nn.Module):
    def __init__(self, channel, ratio):
        super(SEBlock, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channel, channel // ratio, bias=False),
            nn.SiLU(),  # ReLU 대신 SiLU 사용
            nn.Linear(channel // ratio, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, input):
        batch, channel, _, _ = input.size()
        x = self.squeeze(input).view(batch, channel)
        x = self.excitation(x).view(batch, channel, 1, 1)
        x = input * x.expand_as(input)  # scaling
        return x


class MBConv1(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride, padding, ratio=4):
        super(MBConv1, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel, stride=stride, padding=padding,
                                   bias=False, groups=in_channels)
        self.bn = nn.BatchNorm2d(in_channels, momentum=0.99)
        self.SiLU = SiLU()
        self.SEBlock = SEBlock(in_channels, ratio=4)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
        self.bn2 = nn.BatchNorm2d(out_channels, momentum=0.99)
        self.stochastic_depth = StochasticDepth(p=0.2, mode='batch')

    def forward(self, x):
        identity = x
        x = self.depthwise(x)
        x = self.bn(x)
        x = self.SiLU(x)
        x = self.SEBlock(x)
        x = self.conv(x)
        x = self.bn2(x)
        if self.in_channels == self.out_channels and self.stride == 1:
            x = self.stochastic_depth(x)
            x += identity
        return x


class MBConv6(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride, padding, ratio=6):
        super(MBConv6, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.conv1 = nn.Conv2d(in_channels, in_channels * ratio, kernel_size=1, stride=1)
        self.bn = nn.BatchNorm2d(in_channels * ratio, momentum=0.99)
        self.SiLU = SiLU()
        self.depthwise = nn.Conv2d(in_channels * ratio, in_channels * ratio, kernel_size=kernel, stride=stride,
                                   padding=padding, groups=in_channels)
        self.SEBlock = SEBlock(in_channels * ratio, ratio=4)
        self.conv2 = nn.Conv2d(in_channels * ratio, out_channels, kernel_size=1, stride=1)
        self.bn2 = nn.BatchNorm2d(out_channels, momentum=0.99)
        self.stochastic_depth = StochasticDepth(p=0.2, mode='batch')
    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.bn(x)
        x = self.SiLU(x)
        x = self.depthwise(x)
        x = self.SiLU(x)
        x = self.SEBlock(x)
        x = self.conv2(x)
        x = self.bn2(x)
        if self.in_channels == self.out_channels and self.stride == 1:
            x = self.stochastic_depth(x)
            x += identity
        return x

class EfficientNetB0(nn.Module):
    def __init__(self, in_channels=3, num_classes=1000):
        super(EfficientNetB0, self).__init__()
        # Conv3x3, 224x224, 32, 1
        self.stage1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32, momentum=0.99),
            SiLU()
        )
        # MBConv1, k3x3, 112x112, 16, 1
        self.stage2 = MBConv1(in_channels=32, out_channels=16, kernel=3, stride=2, padding=1)
        # MBConv6, k3x3, 112x112, 24, 1
        self.stage3 = MBConv6(in_channels=16, out_channels=24, kernel=3, stride=1, padding=1)
        # MBConv6, k5x5, 56x56, 40, 2
        self.stage4 = nn.Sequential(
            MBConv6(in_channels=24, out_channels=40, kernel=5, stride=2, padding=2),
            MBConv6(in_channels=40, out_channels=40, kernel=5, stride=1, padding=2)
        )
        # MBConv6, k3x3, 28x28, 80, 3
        self.stage5 = nn.Sequential(
            MBConv6(in_channels=40, out_channels=80, kernel=3, stride=2, padding=1),
            MBConv6(in_channels=80, out_channels=80, kernel=3, stride=1, padding=1),
            MBConv6(in_channels=80, out_channels=80, kernel=3, stride=1, padding=1)
        )
        # MBConv6, k5x5, 14x14, 112, 3
        self.stage6 = nn.Sequential(
            MBConv6(in_channels=80, out_channels=112, kernel=5, stride=2, padding=2),
            MBConv6(in_channels=112, out_channels=112, kernel=5, stride=1, padding=2),
            MBConv6(in_channels=112, out_channels=112, kernel=5, stride=1, padding=2)
        )
        # MBConv6, k5x5, 14x14, 192, 4
        self.stage7 = nn.Sequential(
            MBConv6(in_channels=112, out_channels=192, kernel=5, stride=1, padding=2),
            MBConv6(in_channels=192, out_channels=192, kernel=5, stride=1, padding=2),
            MBConv6(in_channels=192, out_channels=192, kernel=5, stride=1, padding=2),
            MBConv6(in_channels=192, out_channels=192, kernel=5, stride=1, padding=2),
        )
        # MBConv6, k3x3, 7x7, 320, 1
        self.stage8 = MBConv6(in_channels=192, out_channels=320, kernel=3, stride=2, padding=1)
        # Conv1x1 & Pooling & FC, 7x7, 1280, 1
        self.stage9 = nn.Sequential(
            # nn.Conv2d(in_channels=320, out_channels=1280, kernel_size=7, stride=1),
            nn.Conv2d(in_channels=320, out_channels=1280, kernel_size=1, stride=1), # kernel size를 7->1로 수정
            nn.BatchNorm2d(1280, momentum=0.99),
            SiLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(p=0.2),
            nn.Linear(1280, num_classes),
        )
    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x = self.stage6(x)
        x = self.stage7(x)
        x = self.stage8(x)
        x = self.stage9(x)
        return x

model = EfficientNetB0().to(device)
summary(model, (3, 224, 224))

# ----------------------------------------------------------------------------------------------
# 1. 모델 저장 - torch.save(model, os.path.join(logs_dir+'/model', 'model.pth'.format(epochs)))
# 모델 불러오기
#model = torch.load("./logs/model/EfficientNet_CIFAR10/model.pth")


# 2. 가중치 저장 - torch.save(model.state_dict(), os.path.join(logs_dir+'/model', 'model_weights.pth'.format(epochs)))
model = EfficientNetB0()  # 모델 인스턴스 생성
model.load_state_dict(torch.load("./logs/model/EfficientNet_CIFAR10/model_weights.pth"))    # 가중치 불러오기


# 3. 체크포인트 저장
# checkpoint = {
#     'epoch': epochs,
#     'model_state_dict': model.state_dict(),
#     'optimizer_state_dict': optimizer.state_dict(),
#     'loss': loss,
# }
# torch.save(checkpoint, os.path.join(logs_dir+'/model/NASNet_CIFAR10', 'checkpoint.pth'))


# model = NASNet().to(device)  # 모델 인스턴스 생성
# 체크포인트 불러오기
# checkpoint = torch.load("./logs/model/NASNet_CIFAR10/checkpoint.pth")

# 모델 및 옵티마이저 상태 불러오기
# optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
# epoch = checkpoint['epoch']
# loss = checkpoint['loss']
# print(optimizer, epoch, loss)
# # 옵티마이저 불러오기
# epochs = 50
# from torch.optim.lr_scheduler import CosineAnnealingLR
# lr_scheduler = CosineAnnealingLR(optimizer, T_max=epoch)





# --------------------------------------------------------------------------------------
mean = torch.tensor([0.485, 0.456, 0.406], device=torch.device('cuda'))
std = torch.tensor([0.229, 0.224, 0.225], device=torch.device('cuda'))

# ImageNet test dataset 결과 출력
criterion = nn.CrossEntropyLoss()
model.to(device)

model.eval()
test_loss = 0.0
correct = 0
total = 0
topk_correct = {k: 0 for k in (5,)}

with tqdm(total=len(test_loader), unit="batch", ncols=100, desc="Testing") as pbar:
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            for k in topk_correct.keys():
                _, predicted_topk = outputs.topk(k, dim=1, largest=True, sorted=True)
                topk_correct[k] += sum(labels[i] in predicted_topk[i] for i in range(labels.size(0)))

            pbar.update(1)

test_loss = test_loss / len(test_loader.dataset)
test_accuracy = correct / total
topk_accuracies = {k: topk_correct[k] / total for k in topk_correct}

print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
for k, acc in topk_accuracies.items():
    print(f"Top-{k} Test Accuracy: {acc:.4f}")


# -------------------------------------------------------------------
# 새로운 이미지 데이터 classification predict
# # ImageNet
# data_transforms_forTest = transforms.Compose([
#     transforms.Resize((256, 256)),
#     transforms.RandomCrop(224),
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])
# cifar10
data_transforms_forTest = transforms.Compose([
    transforms.Resize((40, 40)),
    transforms.RandomCrop((32, 32)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


from PIL import Image
# 이미지 예측 함수 정의
def predict_image(image_path):
    # 이미지 로드 및 전처리
    test_img = Image.open(image_path).convert('RGB')
    test_img = data_transforms_forTest(test_img)
    test_img = test_img.unsqueeze(0)  # 배치 차원을 추가하여 (1, C, H, W) 형태로 변경
    test_img = test_img.to(device)  # 이미지를 GPU로 이동

    # 모델 예측
    with torch.no_grad():
        outputs = model(test_img)
        softmaxed_outputs = nn.functional.softmax(outputs, dim=1)
        predicted_probabilities, predicted_indices = torch.topk(softmaxed_outputs, k=10, dim=1)
        predicted_probabilities = predicted_probabilities.cpu().numpy().flatten()
        predicted_indices = predicted_indices.cpu().numpy().flatten()
        predicted_classes = [class_names[idx] for idx in predicted_indices]
        return predicted_probabilities, predicted_classes


test_img_path = ("/home/ivpl-d29/dataset/cifar_test/truck.jpg")

# 이미지 예측 수행
predicted_probabilities, predicted_classes = predict_image(test_img_path)

# 결과 출력
num=0
for prob, cls in zip(predicted_probabilities, predicted_classes):
    print(f'[{num}] : {cls} - {prob:.4f}')
    num+=1