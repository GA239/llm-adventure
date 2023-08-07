import json

from dotenv import load_dotenv, find_dotenv

from adventure.rooms import get_room_by_type, NUM_ATTEMPTS_PER_ROOM
from adventure.utils import game_print, game_print_debug

NUMBER_OF_ATTEMPTS = 3


def run_room_loop_with_config(config: dict) -> int:
    """Run the room loop with the given config"""
    game_print_debug(f"Running room loop with config: {config}")
    room = get_room_by_type(config["room_type"])
    if room is None:
        raise ValueError(f"Room type {config['room_type']} is not supported!")
    result = room(room_config=config).loop()
    game_print_debug(f"Room loop result: {result}")
    return result


def calculate_score(attempts: int, number_of_rooms) -> int:
    max_attempts = NUM_ATTEMPTS_PER_ROOM * number_of_rooms
    return int(100 * (1 - (attempts - 1) / (max_attempts - 1)))


def main_loop():
    """read json from map.json and run the game loop for each room"""
    with open("map.json") as f:
        map_json = json.load(f)
        # TODO: config and game version validation

    game_print(f"Wellcome to the {map_json['name']} game!" + "\n")
    game_print("Step into 'Mind Maze' and navigate through rooms, solving challenges "
               "set by mathematicians, programmers, and other experts.\n"
               "Test your wit and knowledge in this text-based CLI adventure.")
    game_print("\n How to play:")
    game_print(f" - You will go through the rooms and you must solve the riddle in each!")
    game_print(f" - You will have {NUM_ATTEMPTS_PER_ROOM} attempts to solve each challenge. ")
    game_print(f" - One replica one attempt!")
    game_print(f" - If you fail to solve the challenge, you will lose 1 hp ( ♥ ).")
    game_print(f" - If you lose all your hp, you will lose the game.")

    game_print(f"\nLet's get started! Good luck!\n")

    hp = NUMBER_OF_ATTEMPTS
    score = 0

    for room_config in map_json["rooms"]:
        for attempt in range(hp):
            room_config["hp"] = hp
            rs = run_room_loop_with_config(room_config)
            if not rs:
                hp -= 1
            else:
                score += rs
                break

            if hp == 0:
                game_print("You lost!")
                return calculate_score(score, len(map_json["rooms"]))

    return calculate_score(hp, len(map_json["rooms"]))


if __name__ == '__main__':
    load_dotenv(find_dotenv(raise_error_if_not_found=True))
    fnl_score = main_loop()
    game_print(f"Your final score is: {fnl_score}/100")
