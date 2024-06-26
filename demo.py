#! /usr/bin/env python3

import tensorflow.lite as tflite
from argparse import ArgumentParser

def main():
    args = ArgumentParser()
    args.add_argument("model")
    args = args.parse_args()

    interpreter = tflite.Interpreter(args.model)
    interpreter.allocate_tensors()

    info_input = interpreter.get_input_details()
    info_output = interpreter.get_output_details()
    
    print(info_input, info_output)

if __name__ == "__main__":
    main()

