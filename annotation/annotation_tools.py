def convert_name_to_unique_id(annotator):
    if "Ayman" in annotator:
        unique_id = 1
    elif "Lag" in annotator:
        unique_id = 2
    elif "Anna" in annotator:
        unique_id = 3
    elif "Abddou" in annotator:
        unique_id = 4
    elif "Emmanuelle" in annotator:
        unique_id = 5
    elif "Hili" in annotator:
        unique_id = 6
    elif "Juliette" in annotator:
        unique_id = 7
    elif "Mahamadou" in annotator:
        unique_id = 8
    elif "Folagnimi" in annotator:
        unique_id = 9
    elif "Jules" in annotator:
        unique_id = 10
    elif "Elvino" in annotator:
        unique_id = 11
    elif "Chaima" in annotator:
        unique_id = 12
    elif "Marc" in annotator:
        unique_id = 13
    elif "Jaouad" in annotator:
        unique_id = 14
    elif "ground_truth" in annotator:
        unique_id = 15
    else:
        raise Exception("Unknown annotator")
    return unique_id
