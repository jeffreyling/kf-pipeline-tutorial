import kfp
import kfp.dsl as dsl
from kfp.v2.dsl import component
from kfp.v2.components.types.type_annotations import Output
from kfp.v2.components.types.artifact_types import ClassificationMetrics

@component(
    packages_to_install=['sklearn'],
    base_image='python:3.9'
)
def iris_sgdclassifier(test_samples_fraction: float, metrics: Output[ClassificationMetrics]):
    from sklearn import datasets, model_selection
    from sklearn.linear_model import SGDClassifier
    from sklearn.metrics import confusion_matrix

    iris_dataset = datasets.load_iris()
    train_x, test_x, train_y, test_y = model_selection.train_test_split(
        iris_dataset['data'], iris_dataset['target'], test_size=test_samples_fraction)


    classifier = SGDClassifier()
    classifier.fit(train_x, train_y)
    predictions = model_selection.cross_val_predict(classifier, train_x, train_y, cv=3)
    metrics.log_confusion_matrix(
        ['Setosa', 'Versicolour', 'Virginica'],
        confusion_matrix(train_y, predictions).tolist() # .tolist() to convert np array to list.
    )

@dsl.pipeline(
    name='metrics-visualization-pipeline')
def metrics_visualization_pipeline():
    iris_sgdclassifier_op = iris_sgdclassifier(test_samples_fraction=0.3)


if __name__ == '__main__':
    kfp.compiler.Compiler(
        mode=kfp.dsl.PipelineExecutionMode.V2_COMPATIBLE
    ).compile(metrics_visualization_pipeline, 'iris_pipeline.yaml')
