names = [
    "Ayman",
    "Lag",
    "Anna",
    "abdou",
    "Emmanuelle",
    "Hili",
    "Juliette",
    "Mahamadou",
    "Folagnimi",
    "Jules",
    "Elvino",
    "Chaima",
    "Marc",
]

ports = [x for x in range(8080, 8080 + len(names))]

for name, port in zip(names, ports):
    db_name = f"frblimp_50_{name}"

    call = (
        f"PRODIGY_PORT={port} python -m prodigy fr_blimp {db_name} "
        f"../datastore/FrBLiMP/fr_blimp_50_sentences.jsonl -F recipe.py"
    )

    url = f"http://15.157.58.162:{port}/?session={name}"
    print(f"Nom: {name} | URL: {url} ")

    print(call)
