names = [
    "Ayman",
    # "Lag", # Doublon de Marc
    "Anna",
    # "Abdou", # Doublon de Mahamadou
    "Emmanuelle",
    "Hili",
    "Juliette",
    "Mahamadou",
    "Folagnimi",
    "Jules",
    "Elvino",
    "Chaima",
    "Marc",
    "Jaouad",
]

dataset_size = "1711"

ports = [x for x in range(8080, 8080 + len(names))]

for name, port in zip(names, ports):
    db_name = f"frblimp_{dataset_size}_{name}"

    call = (
        f"PRODIGY_PORT={port} python -m prodigy fr_blimp {db_name} "
        f"../datastore/QFrBLiMP/unannotated/fr_blimp_{dataset_size}_sentences.jsonl -F recipe.py"
    )

    url = f"http://15.157.58.162:{port}/?session={name}"
    print(f"Nom: {name} | URL: {url} ")

    print(call)
