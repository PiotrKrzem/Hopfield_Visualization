from src.domain.model import Model
from src.argparser import parse_args
from src.visualization import Visualization

# ---------------------------- MAIN PROGRAM ----------------------------------
def main() -> None: 
    arguments = parse_args()
    model = Model.from_args(arguments)
    Visualization(model)

if __name__ == "__main__":
    main()
