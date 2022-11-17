
# CE7490 Project 2: RAID-6 based Distributed Storage System

The codebase implements a distributed storage system based on RAID6 using the Reed-Solomon code.

## Requirements

- python version > 3.6
- python packages: numpy, galois, tqdm

## Running Instruction

1. Install requirements, run:
    ~~~~
    pip3 install requirements.txt
    ~~~~

2. [Optional] Put file to be stored into ./data/input/ folder.

3. Run the code:
    ~~~~

    python main.py [--file=input1.png --n=6 --m=2 --w=8 shift==True]
    ~~~~

4. Select 1 to distribute file data into distributed nodes.

![image1](https://user-images.githubusercontent.com/33649731/202465115-37a8cbee-d9c2-4875-81dc-59b540337c89.png)


5. Select 2 to erase data in some nodes.
![image2](https://user-images.githubusercontent.com/33649731/202465133-9537e489-48fa-4cfd-b1c9-a5628864c8e9.png)


6. Select 3 to restore the data. Check for restoration correctness in the output.
![image3](https://user-images.githubusercontent.com/33649731/202464371-a25c70d6-610b-4471-8a9d-58f8444dacac.png)

7. Select 4 to exit.
