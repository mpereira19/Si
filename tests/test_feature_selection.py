import unittest
import warnings
import sys
import os


try:
    import si
except:
    DIR = os.path.dirname(os.path.realpath(__file__))
    PATH = os.path.join(DIR, '../src/')
    sys.path.insert(0, PATH)


class testVarianceThreshold(unittest.TestCase):

	def setUp(self):
		from si.data import Dataset, feature_selection
		self.filename = "datasets/lr-example1.data"
		self.dataset = Dataset.from_data(self.filename, labeled=True)
		self.vt_test = feature_selection.VarianceThreshold(0)
		self.assertWarns(Warning, feature_selection.VarianceThreshold, -1)

	def test_fit(self):
		self.vt_test.fit(self.dataset)
		self.assertGreater(len(self.vt_test._var), 0, 'Variance thresold fit size is 0')

	def test_transform(self):
		self.vt_test.fit(self.dataset)
		self.vt_transform = self.vt_test.transform(self.dataset)
		self.assertEqual(self.dataset.X.shape, self.vt_transform.X.shape, "Variance Thresold transform and dataset don't have the same size")


class test_f_classification(unittest.TestCase):

	def setUp(self):
		from si.data import Dataset
		self.filename = "datasets/pima.data"
		self.dataset = Dataset.from_data(self.filename, labeled=True)

	def test_f_classification(self):
		from si.data.feature_selection import f_classification as fcl
		f, p = fcl(self.dataset)
		self.assertEqual(f.shape, (8,), "Wrong f shape size! Right shape (8,) in f_classification")
		self.assertEqual(p.shape, (8,), "Wrong p shape size! Right shape (8,) in f_classification")


class test_f_regression(unittest.TestCase):

	def setUp(self):
		from si.data import Dataset
		self.filename = "datasets/pima.data"
		self.dataset = Dataset.from_data(self.filename, labeled=True)

	def test_f_regression(self):
		from si.data.feature_selection import f_regression as fr
		f, p = fr(self.dataset)
		self.assertEqual(f.shape, (8,), "Wrong f shape size! Right shape (8,) in f_regression")
		self.assertEqual(p.shape, (8,), "Wrong p shape size! Right shape (8,) in f_regression")


class test_selectKbest_f_regression(unittest.TestCase):

	def setUp(self):
		from si.data import Dataset
		from si.data.feature_selection import f_regression, selectKbest
		self.filename = "datasets/pima.data"
		self.dataset = Dataset.from_data(self.filename, labeled=True)
		self.selectKbest = selectKbest(5, f_regression)
		self.assertEqual(5, self.selectKbest.k, f"selectKbest doesn't have correct k")

	def test_fit(self):
		self.selectKbest.fit(self.dataset)
		self.assertEqual(8, len(self.selectKbest.f), "F size doesn't correspond to dataset size")
		self.assertEqual(8, len(self.selectKbest.pvalue), "p size doesn't correspond to dataset size")

	def test_transform(self):
		self.selectKbest.fit(self.dataset)
		self.selectKbest_transform = self.selectKbest.transform(self.dataset)
		self.assertEqual(self.dataset.X.shape[0], self.selectKbest_transform.X.shape[0], "Dataset and selectKbest.transform don't have the same shape")
		self.assertEqual(self.selectKbest_transform.X.shape[1], self.selectKbest.k)

	def test_fit_transform(self):
		self.selectKbest_fit_transform = self.selectKbest.fit_transform(self.dataset)
		self.assertEqual(self.selectKbest_fit_transform.X.shape[1], self.selectKbest.k)


class test_selectKbest_f_classification(unittest.TestCase):

	def setUp(self):
		from si.data import Dataset
		from si.data.feature_selection import f_classification, selectKbest
		self.filename = "datasets/pima.data"
		self.dataset = Dataset.from_data(self.filename, labeled=True)
		self.selectKbest = selectKbest(5, f_classification)
		self.assertEqual(5, self.selectKbest.k, "selectKbest doesn't have correct k")

	def test_fit(self):
		self.selectKbest.fit(self.dataset)
		self.assertEqual(8, len(self.selectKbest.f), "F size doesn't correspond to dataset size")
		self.assertEqual(8, len(self.selectKbest.pvalue), "p size doesn't correspond to dataset size")

	def test_transform(self):
		self.selectKbest.fit(self.dataset)
		self.selectKbest_transform = self.selectKbest.transform(self.dataset)
		self.assertEqual(self.dataset.X.shape[0], self.selectKbest_transform.X.shape[0], "Dataset and selectKbest.transform don't have the same shape")
		self.assertEqual(self.selectKbest_transform.X.shape[1], self.selectKbest.k)

	def test_fit_transform(self):
		self.selectKbest_fit_transform = self.selectKbest.fit_transform(self.dataset)
		self.assertEqual(self.selectKbest_fit_transform.X.shape[1], self.selectKbest.k)

if __name__ == '__main__':
    unittest.main()
