import os
import unittest
import shutil
from fastapi.testclient import TestClient

# Core imports
from utils.config import AppConfig, load_config
from evaluation.evaluator import GeotechnicalEvaluator
from models.segformer_wrapper import SegFormerWrapper
from analysis_engine.analysis_wrapper import AnalysisEngineWrapper
from prompts.builder import PromptBuilder
from backend.main import app

class TestConfig(unittest.TestCase):
    """Verifies configurations load correctly and apply schema defaults."""
    
    def test_default_config(self):
        config = AppConfig()
        self.assertEqual(config.server.port, 8000)
        self.assertEqual(config.llm.provider, "mock")
        self.assertEqual(config.paths.data_dir, "data")
        
    def test_load_config_fallback(self):
        config = load_config("nonexistent_path.yaml")
        self.assertEqual(config.server.host, "127.0.0.1")

class TestEvaluator(unittest.TestCase):
    """Verifies geohazard text metrics calculations."""
    
    def setUp(self):
        self.evaluator = GeotechnicalEvaluator()
        self.ref = "Landslide core failure probability is critical. Evacuation recommended."
        self.cand = "Landslide core failure probability is critical. Evacuation recommended."
        self.bad_cand = "The weather is very sunny today. No geohazards detected."

    def test_exact_match(self):
        self.assertEqual(self.evaluator.calculate_exact_match(self.ref, self.cand), 1.0)
        self.assertEqual(self.evaluator.calculate_exact_match(self.ref, self.bad_cand), 0.0)

    def test_bleu(self):
        score_perfect = self.evaluator.calculate_bleu(self.ref, self.cand)
        score_bad = self.evaluator.calculate_bleu(self.ref, self.bad_cand)
        self.assertAlmostEqual(score_perfect, 1.0, places=2)
        self.assertEqual(score_bad, 0.0)

    def test_rouge(self):
        scores = self.evaluator.calculate_rouge(self.ref, self.cand)
        self.assertGreaterEqual(scores["rouge1"], 0.9)
        self.assertGreaterEqual(scores["rougel"], 0.9)

    def test_cosine_similarity(self):
        sim = self.evaluator.calculate_semantic_similarity(self.ref, self.cand)
        self.assertAlmostEqual(sim, 1.0, places=4)
        
        sim_bad = self.evaluator.calculate_semantic_similarity(self.ref, self.bad_cand)
        self.assertLess(sim_bad, 0.4)

class TestWrappers(unittest.TestCase):
    """Verifies SegFormer model and Analysis Engine wrappers hook up correctly."""
    
    def setUp(self):
        self.temp_dir = "data/test_temp"
        os.makedirs(self.temp_dir, exist_ok=True)
        self.segformer = SegFormerWrapper("models/segformer_best.pth", "cpu")
        self.analysis = AnalysisEngineWrapper(self.temp_dir)

    def tearDown(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_segformer_predict(self):
        """SegFormerWrapper falls back gracefully when TIFFs are missing."""
        pred_mask, prob_map = self.segformer.predict(
            "nonexistent_coherence.tif",
            "nonexistent_phase.tif"
        )
        self.assertIsNotNone(pred_mask)
        self.assertIsNotNone(prob_map)
        self.assertEqual(pred_mask.shape, prob_map.shape)
        
    def test_analysis_extraction(self):
        """AnalysisEngineWrapper.extract_features() returns expected keys."""
        import numpy as np
        pred_tensor = np.zeros((100, 100), dtype=np.float32)
        features = self.analysis.extract_features(pred_tensor)
        # With real engine, at minimum coherence_path and phase_path will be empty,
        # so check for sample_id and analysis_date which are always set by MetadataExtractor
        self.assertIn("sample_id", features)
        self.assertIn("analysis_date", features)

class TestPrompts(unittest.TestCase):
    """Validates Jinja prompt builder rendering checks."""
    
    def setUp(self):
        # Create temporary templates file for unit test
        self.temp_yaml = "prompts/test_templates.yaml"
        templates = {
            "test_report": "Site: {{ json_data.name }}, Risk: {{ json_data.risk }}",
            "question_answering": "Context: {{ report_content }}, History: {{ chat_history }}, Query: {{ user_query }}"
        }
        with open(self.temp_yaml, "w", encoding="utf-8") as f:
            import yaml
            yaml.dump(templates, f)
        self.builder = PromptBuilder(self.temp_yaml)

    def tearDown(self):
        if os.path.exists(self.temp_yaml):
            os.remove(self.temp_yaml)

    def test_builder_rendering(self):
        # QA Prompt render test
        vars_dict = {
            "report_content": "Slide rate is 50mm/yr.",
            "chat_history": "User: hi\nAssistant: hello",
            "user_query": "Is it active?"
        }
        prompt = self.builder.build_prompt("question_answering", vars_dict)
        self.assertIn("Slide rate is 50mm/yr.", prompt)
        self.assertIn("Is it active?", prompt)

    def test_validation_failure(self):
        with self.assertRaises(ValueError):
            self.builder.build_prompt("question_answering", {"user_query": "only query"})

class TestApiEndpoints(unittest.TestCase):
    """Performs HTTP API integrations testing against the FastAPI backend."""
    
    def setUp(self):
        self.client = TestClient(app)

    def test_health_check(self):
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "healthy")

    def test_report_generation(self):
        dummy_metrics = {
            "landslide_metadata": {"site_name": "Test Site"},
            "displacement_metrics": {"mean_velocity_mm_yr": -10.0},
            "hazard_assessment": {"risk_rating": "Medium"}
        }
        response = self.client.post(
            "/api/report?stream=false",
            json={"json_data": dummy_metrics, "style": "professional_report"}
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn("report", response.json())

if __name__ == "__main__":
    unittest.main()
