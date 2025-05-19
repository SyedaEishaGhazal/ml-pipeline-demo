import unittest
from data_preprocessing import load_and_preprocess_data
from train_model import model
from sklearn.metrics import accuracy_score

class TestPipeline(unittest.TestCase):
    def test_data_shape(self):
        X_train, X_test, y_train, y_test = load_and_preprocess_data()
        self.assertEqual(X_train.shape[1], 4)

    def test_model_accuracy(self):
        X_train, X_test, y_train, y_test = load_and_preprocess_data()
        acc = accuracy_score(y_test, model.predict(X_test))
        self.assertGreater(acc, 0.8)

if __name__ == '__main__':
    unittest.main()
