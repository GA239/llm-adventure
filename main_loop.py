import json

from dotenv import load_dotenv, find_dotenv

from adventure.rooms import get_room_by_type
from adventure.utils import game_print, game_print_debug


def run_room_loop_with_config(config: dict):
    """Run the room loop with the given config"""
    game_print_debug(f"Running room loop with config: {config}")
    room = get_room_by_type(config["room_type"])
    if room is None:
        raise ValueError(f"Room type {config['room_type']} is not supported!")
    room(room_config=config).loop()


def main_loop():
    """read json from map.json and run the game loop for each room"""
    with open("map.json") as f:
        map_json = json.load(f)
        # TODO: config and game version validation

    game_print(f"Wellcome to the {map_json['name']} game!" + "\n")
    game_print(f"You will go through the rooms and you must solve the riddle in each!")
    game_print(f"Let's start!")

    for room_config in map_json["rooms"]:
        run_room_loop_with_config(room_config)


if __name__ == '__main__':
    load_dotenv(find_dotenv(raise_error_if_not_found=True))
    main_loop()
