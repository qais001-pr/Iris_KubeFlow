from kfp import dsl


@dsl.component(
    base_image="python:3.11",
    packages_to_install=[
        "scikit-learn",
        "pandas",
        "numpy"
    ]
)
def load_data(
    X_output: dsl.Output[dsl.Dataset],
    y_output: dsl.Output[dsl.Dataset]
):

    import numpy as np
    from sklearn.datasets import load_iris

    print("Installing/using dependencies...")
    print("Loading Iris dataset...")

    iris = load_iris()

    print("Dataset loaded successfully")
    print("X shape:", iris.data.shape)
    print("y shape:", iris.target.shape)

    np.save(
        X_output.path,
        iris.data
    )

    np.save(
        y_output.path,
        iris.target
    )

    print("X dataset saved to:", X_output.path)
    print("Y dataset saved to:", y_output.path)