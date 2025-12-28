import typing
import pathlib
import sys
from ultralytics import YOLO


def confirm_action(prompt):
    while True:
        response = input(f"{prompt} (yes/no): ").strip().lower()
        if response in ['yes', 'y']:
            return True
        elif response in ['no', 'n']:
            return False
        else:
            print("Please enter 'yes' or 'no'.")


def validateModel(modelf:pathlib.Path, dataf:typing.Union[str|pathlib.Path]):
    # Load a model
    #model = YOLO("yolo11n-cls.pt")  # load an official model
    model = YOLO(modelf)  # load a custom model

    # Validate the model
    metrics = model.val(data=dataf, task='classify', verbose=True)  # no arguments needed, dataset and settings remembered
    print(f"\n\tmetrics.top1={metrics.top1}\n")  # top1 accuracy

if __name__ == "__main__":
    print("Usage:\n\t app model-file dataset")
    modelf = "./mou-thynodu_bom_v72.pt"
    if len(sys.argv) > 2:
        modelf = sys.argv[1]
    datasetHome=r'../datasets/'
    datasetName=r"clsBoM_v03test"
    dataf=datasetHome+datasetName
    if len(sys.argv) > 3:
        modelf = sys.argv[2]

    modelf=pathlib.Path(modelf)
    dataf = pathlib.Path(dataf)
    res = confirm_action(f"val: {modelf.stem}, {dataf.name} ?y/n")
    
    if True == res:
        validateModel(modelf, dataf)
