import argparse
from typing import List
import galois
from util import *

parser = argparse.ArgumentParser(description = 'Implementation of RAID6')
parser.add_argument('target', metavar="file", nargs='?', type=str, default='input1.png', help='name of the file to be stored')
parser.add_argument('n', nargs='?', type=int, default=6, help='number of data disks')
parser.add_argument('m', nargs='?', type=int, default=2, help='number of check disks')
parser.add_argument('w', nargs='?', type=int, default=8, help='number of bits in each block')
parser.add_argument('shift', nargs='?', type=bool, default=True, help='If the data is stored in a shifted manner')
args = parser.parse_args()

gf = galois.GF(2**args.w)

def store_file(target:str) -> None:
    data = load_input(target)
    stripe_data = split_data(data, args.n, args.w)
    parity = calc_parity(stripe_data, gf, args.n, args.m)
    
    save_to_nodes(stripe_data, parity, args.shift)

    print(f"Store file {target} across {args.n + args.m} nodes...")
    pass

def corrupt_disk() -> None:
    mode = input("""
Select disks to corrupt:
    1. Randomly select disks to corrput
    2. Manually select disks to corrupt
""")

    if mode == "1":
        erase_list = random_corrupt(args.n, args.m)
    elif mode == "2":
        erase_list = input(f"""
Enter index of {args.m} disks to corrput, i.e., enter \"0 2\" to corrupt disk 0 and 2\n
""")
        erase_list = [int(x) for x in erase_list.split()]

    else:
        print(f"Error: only 1 or 2 allowed, {mode} entered")

    assert_erase_list(erase_list, args.n, args.m)

    clear_disk(erase_list)
        
    print(f"{erase_list} storage nodes corrupted")
    pass

def rebuild_data() -> None:
    corrupted_data, erase_list = load_nodes()
    assert_erase_list(erase_list, args.n, args.m)
    if len(erase_list)==0:
        print("No restoration is needed as no disk is erased")
    else:
        D = restore_data(corrupted_data, erase_list, gf, args.n, args.m, args.shift)
        C = calc_parity(D, gf, args.n, args.m)
        save_to_nodes(D, C, args.shift)
        print(f"Restored erased data in nodes: {erase_list}")

    verify_correctness(args.target, args.shift)
    pass


if __name__ == '__main__':
    print("======RAID6 test code started======")
    while True:

        action = input(
"""
Please select your action:
    1. Store selected file to storage nodes
    2. Determine failure of storage nodes
    3. Rebuild lost redundancy 
    4. Exit
"""
        )

        if action == "1":
            store_file(args.target)

        elif action == "2": 
            corrupt_disk()

        elif action == "3":
            rebuild_data()

        elif action == "4":
            exit()

        else:
            print(f"Error: only 1, 2, 3 or 4 are allowed, {action} entered")





    



