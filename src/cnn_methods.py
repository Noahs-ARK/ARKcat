import random, math
import numpy as np

#assorted snippets of code used by model_cnn, cnn_train, and cnn_eval

#max length of example in minibatch
def get_max_length(list_of_examples):
    max_length = 0
    for line in list_of_examples:
        max_length = max(max_length, len(line))
    return max_length


if __name__ == "__main__": main()
