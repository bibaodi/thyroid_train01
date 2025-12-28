import argparse
from os.path import basename, dirname, join

import tensorflow as tf
from tensorflow.python.tools import freeze_graph, optimize_for_inference_lib

def parse_args():
    """
    Parse the argument and options passed to the program on the command line.
    """
    parser = argparse.ArgumentParser(description='Handles the conversion of trained checkpoint models to frozen and optimized protobuf for detection (MIL) model.')
    parser.add_argument('-c', '--ckpt_num', nargs='?', type=int, help="The checkpoint number to be frozen.")
    parser.add_argument('-m', '--model_dir', nargs='?', type=str, help="The path to the directory holding the input protobuf file.")
    parser.add_argument('-t', '--as_text', nargs='?', type=int, default=0, help="Whether the input protobuf is stored in text form. Aassumed to be False (0) by default.")
   
    # argument and options are now fields of args object    
    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    # Simplified names
    ckpt_num = args.ckpt_num
    model_dir = args.model_dir
    as_text = bool(args.as_text)

    # Parameters for freezing and optimizing the graph
    if as_text:
        model_path_pb = join(model_dir, "graph.pbtxt")
    else:
        model_path_pb = join(model_dir, "graph.pb")

    input_graph_path = model_path_pb
    input_saver_def_path = ""
    input_binary = not as_text
    input_checkpoint_path = join(model_dir, "model-" + str(ckpt_num))
    output_node_names = "mil_output/noisy_and"
    restore_op_name = "save/restore_all"
    filename_tensor_name = "save/Const:0"
    output_frozen_graph_name = join(model_dir, "frozen_graph.pb")
    clear_devices = True
    initializer_nodes = ""


    # Freeze the graph
    freeze_graph.freeze_graph(
        input_graph_path, 
        input_saver_def_path,
        input_binary, 
        input_checkpoint_path, 
        output_node_names,
        restore_op_name, 
        filename_tensor_name,
        output_frozen_graph_name, 
        clear_devices,
        initializer_nodes,
        input_meta_graph=join(model_dir, "model-" + str(ckpt_num) + ".meta") )


if __name__ == '__main__':
    main()
