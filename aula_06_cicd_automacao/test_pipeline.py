"""
Testes simples para o pipeline de CI/CD.
Valida transformers e funções básicas do código de treinamento.
"""

import sys
import os
import unittest
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Adicionar diretório atual ao path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from preprocessing import CategoricalEncoder, FeatureEngineer, MissingValueImputer
from train import load_and_prepare_data, create_pipeline, PARAMS


class TestPreprocessing(unittest.TestCase):
    """Testa transformers customizados."""
    
    def setUp(self):
        """Cria dados de exemplo para testes."""
        self.sample_data = pd.DataFrame({
            'age': [63, 67, 67, 37, 41],
            'sex': [1, 1, 1, 1, 0],
            'cp': [0, 3, 3, 2, 1],
            'trestbps': [145, 160, 120, 130, 130],
            'chol': [233, 286, 229, 250, 204],
            'fbs': [1, 0, 0, 0, 0],
            'restecg': [2, 2, 2, 0, 2],
            'thalch': [150, 108, 129, 187, 172],
            'exang': [0, 1, 1, 0, 0],
            'oldpeak': [2.3, 1.5, 2.6, 3.5, 1.4],
            'slope': [2, 1, 1, 2, 1],
            'ca': [0, 3, 2, 0, 0],
            'thal': [1, 0, 2, 0, 0]
        })
    
    def test_missing_value_imputer(self):
        """Testa imputação de valores faltantes."""
        # Adicionar valores faltantes
        df = self.sample_data.copy()
        df.loc[0, 'age'] = np.nan
        df.loc[1, 'chol'] = np.nan
        
        imputer = MissingValueImputer(
            numeric_cols=['age', 'chol'],
            categorical_cols=['sex', 'cp']
        )
        imputer.fit(df)
        df_transformed = imputer.transform(df)
        
        # Verificar que não há valores faltantes
        self.assertEqual(df_transformed.isnull().sum().sum(), 0)
        print("✓ Test MissingValueImputer: OK")
    
    def test_categorical_encoder(self):
        """Testa encoding categórico."""
        encoder = CategoricalEncoder()
        encoder.fit(self.sample_data)
        df_encoded = encoder.transform(self.sample_data)
        
        # Verificar que colunas categóricas foram expandidas
        self.assertGreater(len(df_encoded.columns), len(self.sample_data.columns))
        
        # Verificar que não há valores categóricos restantes
        categorical_cols = ['sex', 'cp', 'restecg', 'slope', 'thal']
        for col in categorical_cols:
            self.assertNotIn(col, df_encoded.columns)
        
        print("✓ Test CategoricalEncoder: OK")
    
    def test_feature_engineer(self):
        """Testa criação de features derivadas."""
        engineer = FeatureEngineer()
        df_transformed = engineer.fit_transform(self.sample_data)
        
        # Verificar que novas features foram criadas
        expected_features = ['age_squared', 'age_decade', 'cholesterol_to_age']
        for feat in expected_features:
            self.assertIn(feat, df_transformed.columns)
        
        # Verificar que age_squared está correto
        self.assertEqual(df_transformed['age_squared'].iloc[0], 63**2)
        
        print("✓ Test FeatureEngineer: OK")
    
    def test_pipeline_consistency(self):
        """Testa que o pipeline retorna colunas consistentes."""
        encoder = CategoricalEncoder()
        encoder.fit(self.sample_data)
        
        # Transformar conjunto completo
        df1 = encoder.transform(self.sample_data)
        
        # Transformar uma linha
        df2 = encoder.transform(self.sample_data.iloc[[0]])
        
        # Verificar que têm as mesmas colunas
        self.assertEqual(list(df1.columns), list(df2.columns))
        
        print("✓ Test Pipeline Consistency: OK")


class TestTrainingScript(unittest.TestCase):
    """Testa funções do script de treinamento."""
    
    def test_load_data(self):
        """Testa carregamento de dados."""
        data_path = '../data/heart_disease_uci.csv'
        if not os.path.exists(data_path):
            self.skipTest(f"Dataset não encontrado: {data_path}")
        
        X, y = load_and_prepare_data(data_path)
        
        # Verificar dimensões
        self.assertGreater(len(X), 0)
        self.assertEqual(len(X), len(y))
        
        # Verificar que target é binário
        self.assertTrue(set(y.unique()).issubset({0, 1}))
        
        # Verificar que colunas de metadados foram removidas
        metadata_cols = ['id', 'dataset', 'num']
        for col in metadata_cols:
            self.assertNotIn(col, X.columns)
        
        print("✓ Test Load Data: OK")
    
    def test_create_pipeline(self):
        """Testa criação do pipeline."""
        numeric_cols = ['age', 'trestbps', 'chol']
        categorical_cols = ['sex', 'cp']
        
        pipeline = create_pipeline(PARAMS, numeric_cols, categorical_cols)
        
        # Verificar que pipeline tem os steps corretos
        step_names = [name for name, _ in pipeline.steps]
        expected_steps = ['imputer', 'categorical_encoding', 'feature_engineering', 'scaler', 'classifier']
        
        self.assertEqual(step_names, expected_steps)
        
        # Verificar que o classificador é RandomForest
        classifier = pipeline.steps[-1][1]
        self.assertIsInstance(classifier, RandomForestClassifier)
        
        print("✓ Test Create Pipeline: OK")
    
    def test_baseline_params(self):
        """Testa que hiperparâmetros baseline estão definidos."""
        self.assertIn('n_estimators', PARAMS)
        self.assertIn('random_state', PARAMS)
        self.assertEqual(PARAMS['random_state'], 42)
        
        print("✓ Test Baseline Params: OK")


class TestDataQuality(unittest.TestCase):
    """Testa qualidade básica dos dados."""
    
    def test_dataset_exists(self):
        """Verifica que dataset existe."""
        data_path = '../data/heart_disease_uci.csv'
        self.assertTrue(os.path.exists(data_path), f"Dataset não encontrado: {data_path}")
        print("✓ Test Dataset Exists: OK")
    
    def test_dataset_format(self):
        """Verifica formato básico do dataset."""
        data_path = '../data/heart_disease_uci.csv'
        if not os.path.exists(data_path):
            self.skipTest(f"Dataset não encontrado: {data_path}")
        
        df = pd.read_csv(data_path)
        
        # Verificar que tem linhas e colunas
        self.assertGreater(len(df), 0)
        self.assertGreater(len(df.columns), 0)
        
        # Verificar que coluna 'num' existe (target original)
        self.assertIn('num', df.columns)
        
        print("✓ Test Dataset Format: OK")


def run_tests():
    """Executa todos os testes."""
    # Criar test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Adicionar testes
    suite.addTests(loader.loadTestsFromTestCase(TestPreprocessing))
    suite.addTests(loader.loadTestsFromTestCase(TestTrainingScript))
    suite.addTests(loader.loadTestsFromTestCase(TestDataQuality))
    
    # Executar testes
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Retornar código de erro se testes falharem
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    exit_code = run_tests()
    sys.exit(exit_code)
