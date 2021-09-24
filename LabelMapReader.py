import os
import sys
import traceback

class LabelMapReader:

  def __init__(self):
    pass

  def read(self, label_map_file):
    id    = None
    name  = None
    items = {}
    classes = []    
    with open(label_map_file, "r") as f:
        for line in f:
            line.replace(" ", "")
            if "id" in line:
                id = int(line.split(":")[1].replace(",", "").strip() )

            elif "name" in line:
                name = line.split(":")[1].replace(",", "").strip()
                name = name.replace("'", "").replace("\"", "")

            if id is not None and name is not None:
                classes.append(name)
                #items[name] = id 2021/09/20
                items[id] = name

                id = None
                name = None

    return items, classes



if __name__ == "__main__":
  label_map = "./projects/BloodCells/train/cells_label_map.pbtxt"

  try:
     reader = LabelMapReader()
     items, classes = reader.read(label_map)
     print(items)
     print(classes)

  except Exception as ex:
    traceback.print_exc()



