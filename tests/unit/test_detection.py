#!/usr/bin/env python3
"""
Test suite for VespAI core detection module

Comprehensive tests for camera management, model loading, and detection processing.
"""

import unittest
import sys
import os
import json
import tempfile
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import cv2
import torch
from datetime import datetime
from collections import deque

# Add src to path for imports
project_root = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, project_root)

from src.vespai.core.config import VespAIConfig
from src.vespai.core.detection import CameraManager, ModelManager, DetectionProcessor, parse_resolution


class TestCameraManager(unittest.TestCase):
    """Test cases for CameraManager class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.camera_manager = CameraManager((1920, 1080))
    
    def test_init(self):
        """Test camera manager initialization"""
        self.assertEqual(self.camera_manager.width, 1920)
        self.assertEqual(self.camera_manager.height, 1080)
        self.assertEqual(self.camera_manager.camera_source, 'auto')
        self.assertEqual(self.camera_manager.camera_image_quality, 'max')
        self.assertIsNone(self.camera_manager.cap)

    def test_apply_camera_image_quality_uses_max_when_configured(self):
        """When set to max, quality control should be set to detected backend maximum."""
        camera_manager = CameraManager((1920, 1080), camera_source='usb', camera_image_quality='max')
        camera_manager.cap = Mock()
        camera_manager.device = '/dev/video8'

        with patch.object(camera_manager, '_get_v4l2_quality_control', return_value={'name': 'jpeg_quality', 'min': 30, 'max': 95}):
            with patch.object(camera_manager, '_set_v4l2_control', return_value=True) as mock_set_control:
                camera_manager._apply_camera_image_quality()

        mock_set_control.assert_called_once_with('/dev/video8', 'jpeg_quality', 95)

    def test_apply_camera_image_quality_maps_percent_to_backend_range(self):
        """Numeric 1-100 quality should map into the backend control range."""
        camera_manager = CameraManager((1920, 1080), camera_source='usb', camera_image_quality='50')
        camera_manager.cap = Mock()
        camera_manager.device = '/dev/video8'

        with patch.object(camera_manager, '_get_v4l2_quality_control', return_value={'name': 'jpeg_quality', 'min': 30, 'max': 90}):
            with patch.object(camera_manager, '_set_v4l2_control', return_value=True) as mock_set_control:
                camera_manager._apply_camera_image_quality()

        mock_set_control.assert_called_once_with('/dev/video8', 'jpeg_quality', 60)
    
    @patch('os.path.exists', return_value=True)
    @patch('src.vespai.core.detection.cv2.VideoCapture')
    def test_initialize_camera_with_video_file(self, mock_video_capture, mock_exists):
        """Test camera initialization with video file"""
        mock_cap = Mock()
        mock_cap.isOpened.return_value = True
        mock_video_capture.return_value = mock_cap
        
        result = self.camera_manager.initialize_camera("test_video.mp4")
        
        self.assertEqual(result, mock_cap)
        mock_video_capture.assert_called_once_with("test_video.mp4")
        mock_exists.assert_called_once_with("test_video.mp4")
    
    @patch('src.vespai.core.detection.cv2.VideoCapture')
    @patch('src.vespai.core.detection.time.sleep')
    def test_initialize_camera_success(self, mock_sleep, mock_video_capture):
        """Test successful camera initialization"""
        mock_cap = Mock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = [1920, 1080, 30]  # width, height, fps
        mock_video_capture.return_value = mock_cap
        
        result = self.camera_manager.initialize_camera()
        
        self.assertEqual(result, mock_cap)
        self.assertTrue(mock_video_capture.called)
        mock_sleep.assert_any_call(0.5)
    
    @patch('src.vespai.core.detection.cv2.VideoCapture')
    def test_initialize_camera_failure(self, mock_video_capture):
        """Test camera initialization failure"""
        mock_cap = Mock()
        mock_cap.isOpened.return_value = False
        mock_video_capture.return_value = mock_cap
        
        with patch.object(self.camera_manager, '_initialize_picamera2', side_effect=RuntimeError('no picamera2')):
            with self.assertRaises(RuntimeError):
                self.camera_manager.initialize_camera()

    @patch('src.vespai.core.detection.time.sleep')
    def test_initialize_camera_auto_falls_back_to_picamera2(self, mock_sleep):
        """Test auto mode falling back to Picamera2 when OpenCV camera open fails."""
        with patch.object(self.camera_manager, '_initialize_opencv_camera', side_effect=RuntimeError('opencv failed')):
            with patch.object(self.camera_manager, '_initialize_picamera2') as mock_picamera2:
                mock_picamera2.side_effect = lambda: setattr(self.camera_manager, 'picam2', Mock())

                result = self.camera_manager.initialize_camera()

        self.assertIs(result, self.camera_manager.picam2)
        mock_sleep.assert_any_call(0.5)

    @patch('src.vespai.core.detection.time.sleep')
    def test_initialize_camera_picamera2_mode(self, mock_sleep):
        """Test explicit Picamera2 camera selection."""
        camera_manager = CameraManager((1920, 1080), camera_source='picamera2')

        with patch.object(camera_manager, '_initialize_opencv_camera') as mock_opencv:
            with patch.object(camera_manager, '_initialize_picamera2') as mock_picamera2:
                mock_picamera2.side_effect = lambda: setattr(camera_manager, 'picam2', Mock())

                result = camera_manager.initialize_camera()

        mock_opencv.assert_not_called()
        self.assertIs(result, camera_manager.picam2)
        mock_sleep.assert_any_call(0.5)

    @patch('src.vespai.core.detection.cv2.VideoCapture')
    @patch('src.vespai.core.detection.time.sleep')
    def test_initialize_camera_usb_mode_skips_picamera2(self, mock_sleep, mock_video_capture):
        """Test explicit USB camera selection keeps using OpenCV only."""
        camera_manager = CameraManager((1920, 1080), camera_source='usb')
        mock_cap = Mock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = [1920, 1080, 30]
        mock_video_capture.return_value = mock_cap

        with patch.object(camera_manager, '_initialize_picamera2') as mock_picamera2:
            result = camera_manager.initialize_camera()

        self.assertEqual(result, mock_cap)
        mock_picamera2.assert_not_called()
        mock_sleep.assert_any_call(0.5)

    @patch('src.vespai.core.detection.time.sleep')
    def test_initialize_camera_auto_prefers_picamera2_before_generic_fallback(self, mock_sleep):
        """Auto mode should try Picamera2 before legacy generic V4L2 probing when no USB camera is found."""
        call_order = []

        with patch.object(self.camera_manager, '_get_preferred_video_nodes', return_value=[]):
            with patch.object(self.camera_manager, '_initialize_picamera2') as mock_picamera2:
                with patch.object(self.camera_manager, '_initialize_opencv_camera') as mock_opencv:
                    def opencv_side_effect(*args, **kwargs):
                        call_order.append(('opencv', kwargs.get('include_legacy_nodes')))
                        raise RuntimeError('opencv failed')

                    def picamera_side_effect():
                        call_order.append(('picamera2', None))
                        self.camera_manager.picam2 = Mock()

                    mock_opencv.side_effect = opencv_side_effect
                    mock_picamera2.side_effect = picamera_side_effect

                    result = self.camera_manager.initialize_camera()

        self.assertIs(result, self.camera_manager.picam2)
        self.assertEqual(call_order, [('picamera2', None)])
        mock_sleep.assert_any_call(0.5)

    def test_get_preferred_video_nodes_omits_legacy_nodes_when_requested(self):
        """Preferred node discovery should exclude Pi legacy video nodes until explicit fallback."""
        with patch.dict('os.environ', {'VESPAI_CAMERA_DEVICE': '/dev/video99'}, clear=False):
            with patch.object(self.camera_manager, '_discover_usb_video_nodes', return_value=['/dev/video11']):
                preferred = self.camera_manager._get_preferred_video_nodes(include_legacy_nodes=False)
                fallback = self.camera_manager._get_preferred_video_nodes(include_legacy_nodes=True)

        self.assertEqual(preferred, ['/dev/video99', '/dev/video11'])
        self.assertEqual(
            fallback,
            ['/dev/video99', '/dev/video11', '/dev/video0', '/dev/video8', '/dev/video23', '/dev/video24', '/dev/video25', '/dev/video26'],
        )
    
    def test_read_frame_no_camera(self):
        """Test reading frame without initialized camera"""
        success, frame = self.camera_manager.read_frame()
        
        self.assertFalse(success)
        self.assertIsNone(frame)
    
    def test_read_frame_with_camera(self):
        """Test reading frame with initialized camera"""
        mock_cap = Mock()
        mock_cap.read.return_value = (True, np.zeros((480, 640, 3)))
        self.camera_manager.cap = mock_cap
        
        success, frame = self.camera_manager.read_frame()
        
        self.assertTrue(success)
        self.assertIsNotNone(frame)
    
    def test_release_camera(self):
        """Test camera release"""
        mock_cap = Mock()
        self.camera_manager.cap = mock_cap
        
        self.camera_manager.release()
        
        mock_cap.release.assert_called_once()

    def test_release_picamera2(self):
        """Test Picamera2 resource release."""
        mock_picam2 = Mock()
        self.camera_manager.picam2 = mock_picam2

        self.camera_manager.release()

        mock_picam2.stop.assert_called_once()
        mock_picam2.close.assert_called_once()

    def test_initialize_camera_uses_exact_directory_without_auto_resolve(self):
        """Image dataset path should be used exactly as provided."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yolo_root = os.path.join(tmpdir, 'dataset.yolo')
            test_images_dir = os.path.join(yolo_root, 'images', 'test')
            os.makedirs(test_images_dir, exist_ok=True)

            # Put one image at root and one inside test/images to ensure root is used.
            os.makedirs(yolo_root, exist_ok=True)
            root_image = np.zeros((16, 16, 3), dtype=np.uint8)
            root_image_path = os.path.join(yolo_root, 'root.jpg')
            cv2.imwrite(root_image_path, root_image)

            sample_image = np.zeros((16, 16, 3), dtype=np.uint8)
            image_path = os.path.join(test_images_dir, 'sample.jpg')
            cv2.imwrite(image_path, sample_image)

            self.camera_manager.initialize_camera(yolo_root)
            self.assertTrue(self.camera_manager.image_sequence_mode)
            self.assertEqual(self.camera_manager.device, f"image_dir:{yolo_root}")
            self.assertEqual(len(self.camera_manager.image_files), 1)
            self.assertTrue(self.camera_manager.image_files[0].endswith('root.jpg'))


class TestModelManager(unittest.TestCase):
    """Test cases for ModelManager class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.model_path = "test_model.pt"
        self.model_manager = ModelManager(self.model_path, confidence=0.8)
    
    def test_init(self):
        """Test model manager initialization"""
        self.assertEqual(self.model_manager.model_path, "test_model.pt")
        self.assertEqual(self.model_manager.confidence, 0.8)
        self.assertIsNone(self.model_manager.model)
        self.assertEqual(self.model_manager.class_names, {})
    
    @patch('os.path.exists')
    def test_find_model_file_exists(self, mock_exists):
        """Test finding existing model file"""
        mock_exists.return_value = True
        
        result = self.model_manager._find_model_file()
        
        self.assertTrue(result)
        self.assertEqual(self.model_manager.model_path, "test_model.pt")
    
    @patch('os.path.exists')
    def test_find_model_file_fallback(self, mock_exists):
        """Test finding model file with fallback paths"""
        # First call (original path) returns False, second call (fallback) returns True
        mock_exists.side_effect = [False, True]
        
        result = self.model_manager._find_model_file()
        
        self.assertTrue(result)
        # Should update to the first current fallback path
        self.assertTrue(
            self.model_manager.model_path.endswith(
                "models/L4-YOLOV26-asianhornet_2026-03-13_08-57-52.onnx"
            )
        )
    
    @patch('os.path.exists')
    def test_find_model_file_not_found(self, mock_exists):
        """Test model file not found anywhere"""
        mock_exists.return_value = False
        
        result = self.model_manager._find_model_file()
        
        self.assertFalse(result)
    
    @patch.object(ModelManager, '_find_model_file')
    @patch.object(ModelManager, '_load_via_yolov5_package')
    def test_load_model_success(self, mock_load, mock_find):
        """Test successful model loading"""
        mock_find.return_value = True
        mock_model = Mock()
        mock_model.names = {0: 'crabro', 1: 'velutina'}
        mock_load.return_value = mock_model
        mock_load.__name__ = '_load_via_yolov5_package'  # Fix __name__ issue
        
        result = self.model_manager.load_model()
        
        self.assertEqual(result, mock_model)
        self.assertEqual(self.model_manager.model, mock_model)
        self.assertEqual(mock_model.conf, 0.8)
    
    @patch.object(ModelManager, '_find_model_file')
    def test_load_model_file_not_found(self, mock_find):
        """Test model loading when file not found"""
        mock_find.return_value = False
        
        with self.assertRaises(RuntimeError) as context:
            self.model_manager.load_model()
        
        self.assertIn("Model file not found", str(context.exception))
    
    def test_predict_no_model(self):
        """Test prediction without loaded model"""
        frame = np.zeros((480, 640, 3))
        
        with self.assertRaises(RuntimeError):
            self.model_manager.predict(frame)
    
    def test_predict_with_model(self):
        """Test prediction with loaded model"""
        mock_model = Mock()
        mock_predictions = Mock()
        mock_model.return_value = mock_predictions
        self.model_manager.model = mock_model
        
        frame = np.zeros((480, 640, 3), dtype=np.uint8)  # Fix dtype
        result = self.model_manager.predict(frame)
        
        self.assertEqual(result, mock_predictions)
        mock_model.assert_called_once()

    def test_load_sidecar_class_names(self):
        """Test loading class names from a model sidecar metadata file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, 'test-model.onnx')
            metadata_path = os.path.join(tmpdir, 'test-model_metadata.json')
            with open(model_path, 'wb') as handle:
                handle.write(b'')
            with open(metadata_path, 'w', encoding='utf-8') as handle:
                json.dump({'names': {0: 'Bee', 1: 'Vespa-Crabro', 2: 'Vespa-Velutina', 3: 'Wasp'}}, handle)

            manager = ModelManager(model_path, confidence=0.8)
            self.assertEqual(
                manager._load_sidecar_class_names(),
                {0: 'Bee', 1: 'Vespa-Crabro', 2: 'Vespa-Velutina', 3: 'Wasp'},
            )

    def test_decode_ncnn_output_packed_rows(self):
        """NCNN packed output rows should decode into standard prediction tuples."""
        manager = ModelManager("dummy_ncnn_model", confidence=0.6)
        manager.class_names = {0: 'Bee', 1: 'Vespa-Crabro', 2: 'Vespa-Velutina', 3: 'Wasp'}

        # Format: [cx, cy, w, h, score_class0..score_class3]
        packed = np.array([
            [200.0, 220.0, 100.0, 80.0, 0.05, 0.10, 0.91, 0.02],
            [320.0, 300.0, 120.0, 90.0, 0.80, 0.05, 0.03, 0.01],
        ], dtype=np.float32)

        preds, top_pred, top_class_id, top_class_conf, debug_summary = manager._decode_ncnn_output(
            packed,
            image_height=512,
            image_width=512,
            conf_threshold=0.6,
        )

        self.assertEqual(len(preds), 1)
        self.assertIsNotNone(top_pred)
        self.assertEqual(top_class_id, 2)
        self.assertAlmostEqual(top_class_conf, 0.91, places=4)
        self.assertIn('Vespa-Velutina', debug_summary)

    def test_decode_ncnn_output_uses_class_aware_nms(self):
        """Overlapping boxes from different classes should both survive NMS."""
        manager = ModelManager("dummy_ncnn_model", confidence=0.5)
        manager.class_names = {0: 'Bee', 1: 'Vespa-Crabro', 2: 'Vespa-Velutina', 3: 'Wasp'}

        packed = np.array([
            [256.0, 256.0, 120.0, 120.0, 0.01, 0.98, 0.02, 0.01],
            [256.0, 256.0, 118.0, 118.0, 0.01, 0.01, 0.97, 0.01],
        ], dtype=np.float32)

        preds, _, _, _, _ = manager._decode_ncnn_output(
            packed,
            image_height=512,
            image_width=512,
            conf_threshold=0.5,
        )

        self.assertEqual(len(preds), 2)
        classes = sorted(int(item[5]) for item in preds)
        self.assertEqual(classes, [1, 2])

    def test_decode_ncnn_output_transposed_layout(self):
        """NCNN outputs shaped [features, anchors] should be transposed before decoding."""
        manager = ModelManager("dummy_ncnn_model", confidence=0.5)
        manager.class_names = {0: 'Bee', 1: 'Vespa-Crabro', 2: 'Vespa-Velutina', 3: 'Wasp'}

        # Start with [anchors, features] and transpose to [features, anchors].
        anchors_first = np.array([
            [256.0, 256.0, 120.0, 120.0, 0.01, 0.97, 0.01, 0.01],
            [280.0, 260.0, 110.0, 100.0, 0.01, 0.02, 0.96, 0.01],
        ], dtype=np.float32)
        transposed = anchors_first.T

        preds, top_pred, top_class_id, top_class_conf, _ = manager._decode_ncnn_output(
            transposed,
            image_height=512,
            image_width=512,
            conf_threshold=0.01,
        )

        self.assertEqual(len(preds), 2)
        self.assertIsNotNone(top_pred)
        self.assertIn(top_class_id, (1, 2))
        self.assertGreaterEqual(top_class_conf, 0.96)


class TestDetectionProcessor(unittest.TestCase):
    """Test cases for DetectionProcessor class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.processor = DetectionProcessor()
    
    def test_init(self):
        """Test detection processor initialization"""
        self.assertEqual(self.processor.stats["total_velutina"], 0)
        self.assertEqual(self.processor.stats["total_crabro"], 0)
        self.assertEqual(self.processor.stats["total_detections"], 0)
        self.assertIsInstance(self.processor.stats["detection_log"], deque)
        self.assertEqual(len(self.processor.hourly_detections), 24)
    
    def test_process_detections_no_detections(self):
        """Test processing frame with no detections"""
        # Mock empty results
        mock_results = Mock()
        mock_results.pred = [torch.tensor([])]
        
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        velutina, crabro, annotated = self.processor.process_detections(
            mock_results, frame, 1
        )
        
        self.assertEqual(velutina, 0)
        self.assertEqual(crabro, 0)
        np.testing.assert_array_equal(annotated, frame)
    
    def test_process_detections_with_velutina(self):
        """Test processing frame with Asian hornet detection"""
        import torch
        
        # Mock results with one velutina detection
        mock_results = Mock()
        mock_results.pred = [torch.tensor([[100, 100, 200, 200, 0.95, 1]])]  # cls=1 is velutina
        
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        velutina, crabro, annotated = self.processor.process_detections(
            mock_results, frame, 1
        )
        
        self.assertEqual(velutina, 1)
        self.assertEqual(crabro, 0)
        self.assertEqual(self.processor.stats["total_velutina"], 1)
        self.assertEqual(self.processor.stats["total_detections"], 1)
        
        # Check that detection was logged
        self.assertEqual(len(self.processor.stats["detection_log"]), 1)
        log_entry = self.processor.stats["detection_log"][0]
        self.assertEqual(log_entry["species"], "velutina")
    
    def test_process_detections_with_crabro(self):
        """Test processing frame with European hornet detection"""
        import torch
        
        # Mock results with one crabro detection
        mock_results = Mock()
        mock_results.pred = [torch.tensor([[150, 150, 250, 250, 0.87, 0]])]  # cls=0 is crabro
        
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        velutina, crabro, annotated = self.processor.process_detections(
            mock_results, frame, 2
        )
        
        self.assertEqual(velutina, 0)
        self.assertEqual(crabro, 1)
        self.assertEqual(self.processor.stats["total_crabro"], 1)
        self.assertEqual(self.processor.stats["total_detections"], 1)
        
        # Check that detection was logged
        self.assertEqual(len(self.processor.stats["detection_log"]), 1)
        log_entry = self.processor.stats["detection_log"][0]
        self.assertEqual(log_entry["species"], "crabro")
    
    def test_process_detections_low_confidence(self):
        """Test processing detections below confidence threshold"""
        import torch
        
        # Mock results with low confidence detection
        mock_results = Mock()
        mock_results.pred = [torch.tensor([[100, 100, 200, 200, 0.3, 1]])]  # Low confidence
        
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        velutina, crabro, annotated = self.processor.process_detections(
            mock_results, frame, 1, confidence_threshold=0.8
        )
        
        # Should be filtered out due to low confidence
        self.assertEqual(velutina, 0)
        self.assertEqual(crabro, 0)
        self.assertEqual(self.processor.stats["total_detections"], 0)

    def test_process_detections_logs_frame_prediction_for_yolov5_output(self):
        """Dataset frame logging should still work for YOLOv5/YOLOv8-style outputs."""
        import torch

        mock_results = Mock()
        mock_results.pred = [torch.tensor([[120, 130, 220, 230, 0.55, 1]])]

        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        velutina, crabro, _ = self.processor.process_detections(
            mock_results,
            frame,
            7,
            confidence_threshold=0.8,
            log_frame_prediction=True,
        )

        self.assertEqual(velutina, 0)
        self.assertEqual(crabro, 0)
        self.assertEqual(len(self.processor.stats["detection_log"]), 1)
        log_entry = self.processor.stats["detection_log"][0]
        self.assertEqual(log_entry["class_id"], 1)
        self.assertEqual(log_entry["species"], "velutina")

    def test_process_detections_skips_very_low_frame_prediction_log(self):
        """Very low-confidence frame predictions should be omitted from fallback logs."""
        import torch

        mock_results = Mock()
        mock_results.pred = [torch.tensor([[120, 130, 220, 230, 0.10, 1]])]

        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        self.processor.process_detections(
            mock_results,
            frame,
            8,
            confidence_threshold=0.8,
            log_frame_prediction=True,
        )

        self.assertEqual(len(self.processor.stats["detection_log"]), 0)
    
    def test_process_detections_multiple_hornets(self):
        """Test processing multiple hornet detections"""
        import torch
        
        # Mock results with multiple detections
        detections = torch.tensor([
            [100, 100, 200, 200, 0.95, 1],  # Velutina
            [300, 300, 400, 400, 0.89, 0],  # Crabro
            [500, 500, 600, 600, 0.92, 1],  # Another Velutina
        ])
        mock_results = Mock()
        mock_results.pred = [detections]
        
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        velutina, crabro, annotated = self.processor.process_detections(
            mock_results, frame, 3
        )
        
        self.assertEqual(velutina, 2)
        self.assertEqual(crabro, 1)
        self.assertEqual(self.processor.stats["total_detections"], 3)
    
    def test_detection_frame_storage_limit(self):
        """Test that detection frame storage is limited"""
        import torch
        
        # Create many detections to test frame limit
        mock_results = Mock()
        mock_results.pred = [torch.tensor([[100, 100, 200, 200, 0.95, 1]])]
        
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Process many frames to exceed storage limit
        for i in range(25):  # More than the 20 frame limit
            self.processor.process_detections(mock_results, frame, i + 1)
        
        # Should not exceed 20 stored frames
        self.assertLessEqual(len(self.processor.stats["detection_frames"]), 20)

    def test_conflicting_class_map_override_is_ignored_for_explicit_labels(self):
        """Explicit model labels should win over contradictory overrides."""
        import torch

        self.processor.set_class_names(
            {0: 'Bee', 1: 'Bee', 2: 'Vespa-Crabro', 3: 'Vespa-Crabro'},
            '1:crabro,2:velutina',
        )

        mock_results = Mock()
        mock_results.pred = [torch.tensor([[100, 100, 200, 200, 0.91, 1]])]
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        velutina, crabro, _ = self.processor.process_detections(
            mock_results,
            frame,
            1,
            confidence_threshold=0.8,
        )

        self.assertEqual(velutina, 0)
        self.assertEqual(crabro, 0)
        self.assertEqual(self.processor.stats['total_bee'], 1)


class TestUtilityFunctions(unittest.TestCase):
    """Test cases for utility functions"""
    
    def test_parse_resolution_standard_formats(self):
        """Test parsing standard resolution formats"""
        self.assertEqual(parse_resolution("4k"), (3840, 2160))
        self.assertEqual(parse_resolution("1080p"), (1920, 1080))
        self.assertEqual(parse_resolution("720p"), (1280, 720))
    
    def test_parse_resolution_custom_format(self):
        """Test parsing custom resolution format"""
        self.assertEqual(parse_resolution("1600x900"), (1600, 900))
        self.assertEqual(parse_resolution("800x600"), (800, 600))
    
    def test_parse_resolution_invalid_format(self):
        """Test parsing invalid resolution format"""
        # Should return default resolution
        self.assertEqual(parse_resolution("invalid"), (1920, 1080))
        self.assertEqual(parse_resolution("800"), (1920, 1080))
        self.assertEqual(parse_resolution(""), (1920, 1080))


class TestConfig(unittest.TestCase):
    """Test cases for configuration defaults and normalization."""

    def test_camera_source_defaults_to_auto(self):
        """Camera source should default to auto when not specified."""
        config = VespAIConfig()
        config.parse_args([])

        self.assertEqual(config.get('camera_source'), 'auto')

    def test_camera_source_picamera3_alias_normalizes(self):
        """Camera Module 3 alias should map to the Picamera2 backend."""
        config = VespAIConfig()
        config.parse_args(['--camera-source', 'picamera3'])

        self.assertEqual(config.get('camera_source'), 'picamera2')

    def test_camera_image_quality_defaults_to_max(self):
        """Camera image quality should default to max."""
        config = VespAIConfig()
        config.parse_args([])

        self.assertEqual(config.get('camera_image_quality'), 'max')

    def test_camera_image_quality_accepts_numeric_values(self):
        """Numeric camera image quality should be preserved after validation."""
        config = VespAIConfig()
        config.parse_args(['--camera-image-quality', '72'])
        config.validate()

        self.assertEqual(config.get('camera_image_quality'), '72')

    def test_camera_image_quality_rejects_out_of_range_values(self):
        """Camera image quality must be in range 1-100 when numeric."""
        config = VespAIConfig()
        config.parse_args(['--camera-image-quality', '101'])

        with self.assertRaises(ValueError):
            config.validate()


class TestDetectionIntegration(unittest.TestCase):
    """Integration tests for detection components"""
    
    @patch('src.vespai.core.detection.cv2.VideoCapture')
    @patch('src.vespai.core.detection.time.sleep')
    def test_camera_and_processor_integration(self, mock_sleep, mock_video_capture):
        """Test integration between camera manager and detection processor"""
        # Setup camera mock
        mock_cap = Mock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = [640, 480, 30]
        mock_cap.read.return_value = (True, np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8))
        mock_video_capture.return_value = mock_cap
        
        # Initialize components
        camera = CameraManager((640, 480))
        processor = DetectionProcessor()
        
        # Initialize camera
        cap = camera.initialize_camera()
        self.assertIsNotNone(cap)
        
        # Read frame and verify
        success, frame = camera.read_frame()
        self.assertTrue(success)
        self.assertIsNotNone(frame)
        self.assertEqual(frame.shape, (480, 640, 3))
        
        # Mock detection results
        import torch
        mock_results = Mock()
        mock_results.pred = [torch.tensor([[100, 100, 200, 200, 0.95, 1]])]
        
        # Process detections
        velutina, crabro, annotated = processor.process_detections(mock_results, frame, 1)
        
        self.assertEqual(velutina, 1)
        self.assertEqual(crabro, 0)
        self.assertEqual(processor.stats["total_detections"], 1)


if __name__ == '__main__':
    unittest.main(verbosity=2)