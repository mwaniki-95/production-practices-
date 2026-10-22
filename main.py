"""Run the Iris training workflow."""

from production_iris.data import load_iris_data
from production_iris.logging_config import configure_logging
from production_iris.model import train_model


def main() -> None:
    """Load data, train the model and report its metrics."""
    configure_logging()

    df = load_iris_data()
    model, metrics = train_model(df)

    print(f"Test accuracy: {metrics['accuracy']:.4f}")

    new_flower = df[
        [
            "sepal length (cm)",
            "sepal width (cm)",
            "petal length (cm)",
            "petal width (cm)",
        ]
    ].iloc[[0]]

    prediction = model.predict(new_flower)
    print(f"Predicted class: {int(prediction[0])}")


if __name__ == "__main__":
    main()
