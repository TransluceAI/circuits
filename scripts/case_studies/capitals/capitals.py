"""
Generate circuits for the capitals case study.
"""

import os
from pathlib import Path
from circuits.analysis.circuit_ops import Circuit
from util.subject import Subject, llama31_8B_instruct_config

from env_util import ENV
if ENV.ARTIFACTS_DIR is None:
    raise RuntimeError("ARTIFACTS_DIR must be set in the .env file")

ARTIFACTS_DIR = Path(ENV.ARTIFACTS_DIR) / "capitals_circuit_hypotheses"

city_state_capital = [
    ("Dallas", "Texas", "Austin"),
    ("Birmingham", "Alabama", "Montgomery"),
    ("Anchorage", "Alaska", "Juneau"),
    ("Tucson", "Arizona", "Phoenix"),
    ("Fayetteville", "Arkansas", "Little Rock"),
    ("Los Angeles", "California", "Sacramento"),
    ("Colorado Springs", "Colorado", "Denver"),
    ("Bridgeport", "Connecticut", "Hartford"),
    ("Wilmington", "Delaware", "Dover"),
    ("Jacksonville", "Florida", "Tallahassee"),
    ("Savannah", "Georgia", "Atlanta"),
    ("Hilo", "Hawaii", "Honolulu"),
    ("Idaho Falls", "Idaho", "Boise"),
    ("Chicago", "Illinois", "Springfield"),
    ("Fort Wayne", "Indiana", "Indianapolis"),
    ("Cedar Rapids", "Iowa", "Des Moines"),
    ("Wichita", "Kansas", "Topeka"),
    ("Louisville", "Kentucky", "Frankfort"),
    ("New Orleans", "Louisiana", "Baton Rouge"),
    ("Portland", "Maine", "Augusta"),
    ("Baltimore", "Maryland", "Annapolis"),
    ("Worcester", "Massachusetts", "Boston"),
    ("Detroit", "Michigan", "Lansing"),
    ("Minneapolis", "Minnesota", "Saint Paul"),
    ("Gulfport", "Mississippi", "Jackson"),
    ("St. Louis", "Missouri", "Jefferson City"),
    ("Billings", "Montana", "Helena"),
    ("Omaha", "Nebraska", "Lincoln"),
    ("Las Vegas", "Nevada", "Carson City"),
    ("Manchester", "New Hampshire", "Concord"),
    ("Newark", "New Jersey", "Trenton"),
    ("Albuquerque", "New Mexico", "Santa Fe"),
    ("New York City", "New York", "Albany"),
    ("Charlotte", "North Carolina", "Raleigh"),
    ("Fargo", "North Dakota", "Bismarck"),
    ("Cleveland", "Ohio", "Columbus"),
    ("Tulsa", "Oklahoma", "Oklahoma City"),
    ("Portland", "Oregon", "Salem"),
    ("Philadelphia", "Pennsylvania", "Harrisburg"),
    ("Warwick", "Rhode Island", "Providence"),
    ("Charleston", "South Carolina", "Columbia"),
    ("Sioux Falls", "South Dakota", "Pierre"),
    ("Memphis", "Tennessee", "Nashville"),
    ("Provo", "Utah", "Salt Lake City"),
    ("Burlington", "Vermont", "Montpelier"),
    ("Virginia Beach", "Virginia", "Richmond"),
    ("Seattle", "Washington", "Olympia"),
    ("Huntington", "West Virginia", "Charleston"),
    ("Milwaukee", "Wisconsin", "Madison"),
    ("Casper", "Wyoming", "Cheyenne"),
]
prompts = [
    f"What is the capital of the state containing {city}?" for city, _, _ in city_state_capital
]
seed_responses = ["Answer:"] * len(prompts)
labels = [" " + capital for _, _, capital in city_state_capital]


def main():
    subject = Subject(llama31_8B_instruct_config)
    subject.tokenizer

    # convert to dataframes
    circuit = Circuit.from_dataset(
        subject,
        prompts,
        seed_responses,
        labels,
        return_nodes_only=False,
        neurons=None,
        percentage_threshold=0.005,
        batch_size=1,
        verbose=False,
        k=5,
        apply_blacklist=True,
    )

    os.makedirs(ARTIFACTS_DIR / "case_studies", exist_ok=True)
    circuit.save_to_pickle(ARTIFACTS_DIR / "case_studies" / "capitals_circuit.pkl")


if __name__ == "__main__":
    main()
