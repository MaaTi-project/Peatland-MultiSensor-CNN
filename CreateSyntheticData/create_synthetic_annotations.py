import random
import os

dictionary_of_coordinates = {
    "minx": 393200.0,
    "maxx": 422800.0,
    "maxy": 7324390.0,
    "miny": 7294650.0,
    "pix_x": 10.0,
    "pix_y": -10.0,
    "height": 2974,
    "width": 2960
}


def create_random_annotation(total_number_of_points, min_class_range, max_class_range, save_path):

    opener = open(save_path, "w")
    opener.write(f"N,E,classes\n")

    for i in range(total_number_of_points):
        N = random.uniform(dictionary_of_coordinates["miny"], dictionary_of_coordinates["maxy"])
        E = random.uniform(dictionary_of_coordinates["minx"], dictionary_of_coordinates["maxx"])
        classes = random.randint(min_class_range, max_class_range)
        opener.write(f"{N},{E},{classes}\n")

    opener.close()

if __name__ == '__main__':

    annotation_path = os.path.join("..", "Annotations", "TestArea", "Dataset")

    if not os.path.exists(annotation_path):
        os.makedirs(annotation_path)

    for types in ['drained', 'undrained']:
        create_random_annotation(total_number_of_points=1000,
                                 min_class_range=1,
                                 max_class_range=36,
                                 save_path=os.path.join(annotation_path, f"annotation_{types}.csv"))
