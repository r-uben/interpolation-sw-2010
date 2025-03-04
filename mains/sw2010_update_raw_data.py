import json
import pandas as pd



def load_sources():
    with open("data/sources.json", "r") as file:
        data = json.load(file)  # json.load() reads directly from a file object
    return data

def main():
    sources = load_sources()
    print(sources)

if __name__ == "__main__":
    main()
