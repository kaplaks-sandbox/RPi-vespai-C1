#!/usr/bin/env python3
"""
VespAI Configuration Module

Handles all configuration management including environment variables,
command line arguments, and default settings.

Author: Jakob Zeise (Zeise Digital)
Modified: Andre Jordaan
Version: 2.0
"""

import os
import argparse
import logging
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
from dotenv import load_dotenv

logger = logging.getLogger(__name__)


class VespAIConfig:
    """
    Central configuration management for VespAI.
    
    Handles environment variables, command line arguments, and default settings
    with proper precedence (CLI args override env vars override defaults).
    """
    
    def __init__(self):
        """Initialize configuration with defaults."""
        # Load environment variables from this repository's .env file and
        # force them to override ambient shell variables from other projects.
        project_root = Path(__file__).resolve().parents[3]
        dotenv_path = project_root / '.env'
        load_dotenv(dotenv_path=dotenv_path, override=True)
        
        # Default configuration
        self.defaults = {
            # Camera settings
            'resolution': '1920x1080',
            'camera_source': 'auto',
            'camera_alias': 'Camera',
            'camera_autofocus': True,
            'camera_image_quality': 'max',
            'camerapi_focus_mode': 'continuous',
            'camerapi_focus_distance_m': 0.0,
            'camerapi_awb_mode': 'auto',
            'camerapi_awb_red_gain': 0.0,
            'camerapi_awb_blue_gain': 0.0,
            'camerapi_color_order': 'bgr',
            'video_file': None,
            'dataset_path': '',

            # Tracking settings
            'tracking_mode': 'off',
            
            # Detection settings  
            'confidence_threshold': 0.8,
            'model_format': 'auto',
            'model_path': 'models/L4-YOLOV26-asianhornet_2026-03-13_08-57-52_ncnn_model',
            'class_map': '',
            'save_detections': True,
            'save_directory': 'data/detections',
            'detection_retention_days': 21,
            'detection_max_file_count': 250,
            'print_detections': False,
            
            # Motion detection
            'enable_motion_detection': True,
            'min_motion_area': 100,
            'dilation_iterations': 1,
            
            # Performance settings
            'frame_delay': 0.1,
            'dataset_frame_delay': 0.6,
            
            # Web interface
            'enable_web': True,
            'web_host': '0.0.0.0',
            'web_port': 5000,
            'web_preview_size': '960x540',
            'live_stream_quality': 72,
            'current_frame_quality': 82,
            'web_color_scale_r': 1.0,
            'web_color_scale_g': 1.0,
            'web_color_scale_b': 1.0,
            
            # SMS settings (disabled by default, use --sms to enable)
            'enable_sms': False,
            'lox24_api_key': '',
            'phone_number': '',
            'lox24_sender': os.getenv('VESPAI_NAME', 'VespAI-C1'),
            'sms_delay_minutes': 5,
            'domain_name': 'localhost',
            'use_https': False,
            
            # Pushover settings (disabled by default, use --push to enable)
            'enable_push': False,
            'pushover_token': '',
            'pushover_user': '',
            'pushover_sender': os.getenv('VESPAI_NAME', 'VespAI-C1'),
            'push_delay_minutes': 5,
            'push_thumbnail': False,
        }
        
        # Current configuration (will be populated from env + args)
        self.config = {}
        self._load_from_environment()

    def _normalize_camera_source(self, value: Any) -> str:
        """Normalize camera source names while keeping user-facing aliases working."""
        normalized = str(value or 'auto').strip().lower()
        if normalized == 'picamera3':
            return 'picamera2'
        return normalized

    def _normalize_camera_alias(self, value: Any) -> str:
        """Normalize user-visible camera alias with a strict 16 character cap."""
        alias = str(value or '').strip()
        if not alias:
            alias = 'Camera'
        if len(alias) > 16:
            alias = alias[:16]
        return alias

    def _normalize_tracking_mode(self, value: Any) -> str:
        """Normalize tracker mode names while keeping aliases working."""
        normalized = str(value or 'off').strip().lower()
        aliases = {
            'none': 'off',
            'false': 'off',
            '0': 'off',
            'simple': 'centroid',
        }
        return aliases.get(normalized, normalized)

    def _normalize_model_format(self, value: Any) -> str:
        """Normalize model format selector for env/CLI overrides."""
        normalized = str(value or 'auto').strip().lower()
        aliases = {
            'none': 'auto',
            'default': 'auto',
        }
        return aliases.get(normalized, normalized)

    def _is_ncnn_dir(self, path: str) -> bool:
        p = Path(path)
        return p.is_dir() and (p / 'model.ncnn.param').exists() and (p / 'model.ncnn.bin').exists()

    def _resolve_model_path_for_format(self):
        """Resolve model_path from model_format preference when possible."""
        model_path_raw = str(self.config.get('model_path') or '').strip()
        if not model_path_raw:
            return

        model_format = self._normalize_model_format(self.config.get('model_format'))
        if model_format == 'auto':
            return

        path = Path(model_path_raw)
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()

        if model_format == 'ncnn':
            if self._is_ncnn_dir(str(path)):
                return

            candidates = []
            if path.suffix.lower() == '.onnx':
                candidates.append(path.with_suffix(''))
            else:
                candidates.append(path)

            for candidate in list(candidates):
                candidates.append(candidate.parent / f"{candidate.name}_ncnn_model")

            for candidate in candidates:
                if self._is_ncnn_dir(str(candidate)):
                    self.config['model_path'] = str(candidate)
                    return

            logger.warning("Model format is 'ncnn' but no NCNN directory found for %s", model_path_raw)
            return

        if model_format == 'onnx':
            if path.suffix.lower() == '.onnx' and path.exists():
                return

            candidates = []
            if self._is_ncnn_dir(str(path)):
                stem = path.name
                if stem.endswith('_ncnn_model'):
                    stem = stem[:-11]
                candidates.append(path.parent / f"{stem}.onnx")
            elif path.suffix:
                candidates.append(path.with_suffix('.onnx'))
            else:
                candidates.append(path.parent / f"{path.name}.onnx")

            for candidate in candidates:
                if candidate.exists():
                    self.config['model_path'] = str(candidate)
                    return

            logger.warning("Model format is 'onnx' but no ONNX file found for %s", model_path_raw)
    
    def _load_from_environment(self):
        """Load configuration from environment variables."""
        # Map environment variable names to config keys
        env_mapping = {
            'RESOLUTION': 'resolution',
            'VESPAI_CAMERA_SOURCE': 'camera_source',
            'VESPAI_CAMERA_ALIAS': 'camera_alias',
            'VESPAI_CAMERAPI_AUTOFOCUS': 'camera_autofocus',
            'CAMERA_IMAGE_QUALITY': 'camera_image_quality',
            'VESPAI_CAMERAPI_FOCUS_MODE': 'camerapi_focus_mode',
            'VESPAI_CAMERAPI_FOCUS_DISTANCE_M': 'camerapi_focus_distance_m',
            'VESPAI_CAMERAPI_AWB_MODE': 'camerapi_awb_mode',
            'VESPAI_CAMERAPI_AWB_RED_GAIN': 'camerapi_awb_red_gain',
            'VESPAI_CAMERAPI_AWB_BLUE_GAIN': 'camerapi_awb_blue_gain',
            'VESPAI_CAMERAPI_COLOR_ORDER': 'camerapi_color_order',
            'CONFIDENCE_THRESHOLD': 'confidence_threshold', 
            'VESPAI_MODEL_FORMAT': 'model_format',
            'MODEL_PATH': 'model_path',
            'VESPAI_CLASS_MAP': 'class_map',
            'VESPAI_DATASET_PATH': 'dataset_path',
            'VESPAI_TRACKING_MODE': 'tracking_mode',
            'SAVE_DETECTIONS': 'save_detections',
            'SAVE_DIRECTORY': 'save_directory',
            'DETECTION_RETENTION_DAYS': 'detection_retention_days',
            'DETECTION_MAX_FILE_COUNT': 'detection_max_file_count',
            'ENABLE_MOTION_DETECTION': 'enable_motion_detection',
            'MIN_MOTION_AREA': 'min_motion_area',
            'FRAME_DELAY': 'frame_delay',
            'DATASET_FRAME_DELAY': 'dataset_frame_delay',
            'ENABLE_WEB': 'enable_web',
            'WEB_HOST': 'web_host',
            'WEB_PORT': 'web_port',
            'WEB_PREVIEW_SIZE': 'web_preview_size',
            'LIVE_STREAM_QUALITY': 'live_stream_quality',
            'CURRENT_FRAME_QUALITY': 'current_frame_quality',
            'WEB_COLOR_SCALE_R': 'web_color_scale_r',
            'WEB_COLOR_SCALE_G': 'web_color_scale_g',
            'WEB_COLOR_SCALE_B': 'web_color_scale_b',
            'ENABLE_SMS': 'enable_sms',
            'LOX24_API_KEY': 'lox24_api_key',
            'PHONE_NUMBER': 'phone_number',
            'LOX24_SENDER': 'lox24_sender',
            'SMS_DELAY_MINUTES': 'sms_delay_minutes',
            'ENABLE_PUSH': 'enable_push',
            'PUSHOVER_TOKEN': 'pushover_token',
            'PUSHOVER_USER': 'pushover_user',
            'PUSHOVER_SENDER': 'pushover_sender',
            'PUSH_DELAY_MINUTES': 'push_delay_minutes',
            'PUSH_THUMBNAIL': 'push_thumbnail',
            'DOMAIN_NAME': 'domain_name',
            'USE_HTTPS': 'use_https',
        }
        
        # Start with defaults
        self.config = self.defaults.copy()
        
        # Override with environment variables
        for env_key, config_key in env_mapping.items():
            env_value = os.getenv(env_key)
            if env_value is not None:
                # Convert types based on default value type
                default_value = self.defaults[config_key]
                try:
                    if isinstance(default_value, bool):
                        self.config[config_key] = env_value.lower() in ('true', '1', 'yes', 'on')
                    elif isinstance(default_value, int):
                        self.config[config_key] = int(env_value)
                    elif isinstance(default_value, float):
                        self.config[config_key] = float(env_value)
                    else:
                        self.config[config_key] = env_value
                        
                    logger.debug("Loaded %s from environment: %s", config_key, env_value)
                except (ValueError, TypeError) as e:
                    logger.warning("Invalid environment value for %s: %s (%s)", env_key, env_value, e)

            self.config['camera_source'] = self._normalize_camera_source(self.config.get('camera_source'))
            self.config['tracking_mode'] = self._normalize_tracking_mode(self.config.get('tracking_mode'))
            self.config['model_format'] = self._normalize_model_format(self.config.get('model_format'))

        self._resolve_model_path_for_format()
    
    def parse_args(self, args=None) -> argparse.Namespace:
        """
        Parse command line arguments and update configuration.
        
        Args:
            args: List of arguments to parse (None for sys.argv)
            
        Returns:
            argparse.Namespace: Parsed arguments
        """
        parser = argparse.ArgumentParser(
            description='VespAI Hornet Detection System',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
        
        # Camera settings
        parser.add_argument('-r', '--resolution', 
                          default=self.config['resolution'],
                          help='Camera resolution (e.g., 1920x1080, 1080p, 720p)')
        parser.add_argument('--camera-source',
                  choices=('auto', 'usb', 'picamera2', 'picamera3'),
                  default=self.config['camera_source'],
                  help='Live camera backend selection (Camera Module 3 uses the Picamera2 backend)')
        parser.add_argument('--camera-alias',
            default=self.config.get('camera_alias', 'Camera'),
            help='Camera display alias shown in logs/UI (max 16 chars)')
        parser.add_argument('--camera-autofocus',
                action='store_true',
                default=self.config['camera_autofocus'],
                help='Enable Picamera2 continuous autofocus (ignored for fixed-focus cameras)')
        parser.add_argument('--camera-image-quality',
              default=self.config['camera_image_quality'],
              help="Camera capture quality: 'max' or integer 1-100 (mapped to backend range when supported)")
        parser.add_argument('--camerapi-focus-mode',
              choices=('continuous', 'auto', 'manual', 'off'),
              default=self.config['camerapi_focus_mode'],
              help='Pi Camera focus mode (Picamera2 only)')
        parser.add_argument('--camerapi-focus-distance-m',
              type=float,
              default=float(self.config.get('camerapi_focus_distance_m', 0.0) or 0.0),
              help='Pi Camera focus distance in meters (e.g. 0.13 for 13 cm, 0 disables)')
        parser.add_argument('--camerapi-awb-mode',
              choices=('auto', 'incandescent', 'tungsten', 'fluorescent', 'indoor', 'daylight', 'cloudy', 'off', 'custom', 'manual'),
              default=self.config['camerapi_awb_mode'],
              help='Pi Camera AWB mode (Picamera2 only)')
        parser.add_argument('--camerapi-awb-red-gain',
              type=float,
              default=float(self.config.get('camerapi_awb_red_gain', 0.0) or 0.0),
              help='Pi Camera manual AWB red gain (set both gains, 0 disables manual gains)')
        parser.add_argument('--camerapi-awb-blue-gain',
              type=float,
              default=float(self.config.get('camerapi_awb_blue_gain', 0.0) or 0.0),
              help='Pi Camera manual AWB blue gain (set both gains, 0 disables manual gains)')
        parser.add_argument('--no-camera-autofocus',
                action='store_true',
                default=False,
                help='Disable Picamera2 autofocus')
        parser.add_argument('-v', '--video',
                          default=self.config['video_file'],
                          help='Video file or image directory to process instead of live camera')

        parser.add_argument('--tracking-mode',
                  choices=('off', 'centroid', 'iou', 'simple'),
                  default=self.config['tracking_mode'],
                  help='Object tracking mode for stable IDs on detections')
        
        # Detection settings
        parser.add_argument('-c', '--conf', '--confidence',
                          type=float,
                          default=self.config['confidence_threshold'], 
                          help='Detection confidence threshold')
        parser.add_argument('--model-path',
                          default=self.config['model_path'],
                          help='Path to model weights or export artifact')
        parser.add_argument('--model-format',
              choices=('auto', 'onnx', 'ncnn'),
              default=self.config['model_format'],
              help='Model runtime format preference (auto, onnx, ncnn)')
        parser.add_argument('--class-map',
                  default=self.config['class_map'],
                  help='Class-to-species mapping, e.g. "0:crabro,1:velutina" or JSON string')
        parser.add_argument('-s', '--save',
                          action='store_true',
                          default=self.config['save_detections'],
                          help='Save detection images')
        parser.add_argument('-sd', '--save-dir',
                          default=self.config['save_directory'],
                          help='Directory to save detection images')
        parser.add_argument('--detection-max-file-count',
                  type=int,
                  default=self.config.get('detection_max_file_count', 250),
                  help='Maximum number of saved detection images to keep (0 disables count cap)')
        parser.add_argument('-p', '--print',
                          action='store_true', 
                          default=self.config['print_detections'],
                          help='Print detection details to console')
        
        # Motion detection
        parser.add_argument('-m', '--motion',
                          action='store_true',
                          default=self.config['enable_motion_detection'],
                          help='Enable motion detection optimization')
        parser.add_argument('-a', '--min-motion-area',
                          type=int,
                          default=self.config['min_motion_area'],
                          help='Minimum motion area threshold')
        parser.add_argument('-d', '--dilation',
                          type=int,
                          default=self.config['dilation_iterations'],
                          help='Dilation iterations for motion detection')
        
        # Performance settings
        parser.add_argument('-b', '--brake',
                          type=float,
                          default=self.config['frame_delay'],
                          help='Frame processing delay in seconds')
        parser.add_argument('--dataset-delay',
                  type=float,
                  default=self.config['dataset_frame_delay'],
                  help='Minimum frame delay for finite dataset inputs (seconds)')
        
        # Web interface
        parser.add_argument('--web',
                          action='store_true',
                          default=self.config['enable_web'],
                          help='Enable web dashboard')
        parser.add_argument('--web-host',
                          default=self.config['web_host'],
                          help='Web server host address')
        parser.add_argument('--web-port',
                          type=int,
                          default=self.config['web_port'],
                          help='Web server port')
        
        # SMS alerts
        parser.add_argument('--sms',
                          action='store_true',
                          default=False,
                          help='Enable SMS alerts (requires LOX24_API_KEY and PHONE_NUMBER)')
        parser.add_argument('--no-sms',
                          action='store_true',
                          default=False,
                          help='Disable SMS alerts')

        # Pushover alerts
        parser.add_argument('--push',
                  action='store_true',
                  default=False,
                  help='Enable Pushover alerts (requires PUSHOVER_TOKEN and PUSHOVER_USER)')
        parser.add_argument('--no-push',
                  action='store_true',
                  default=False,
                  help='Disable Pushover alerts')
        
        # Parse arguments
        parsed_args = parser.parse_args(args)
        
        # Update configuration with parsed arguments
        self._update_from_args(parsed_args)
        self.config['camera_source'] = self._normalize_camera_source(self.config.get('camera_source'))
        self.config['tracking_mode'] = self._normalize_tracking_mode(self.config.get('tracking_mode'))
        self.config['model_format'] = self._normalize_model_format(self.config.get('model_format'))
        self._resolve_model_path_for_format()
        
        return parsed_args
    
    def _update_from_args(self, args: argparse.Namespace):
        """Update configuration from parsed command line arguments."""
        # Map argument attributes to config keys
        arg_mapping = {
            'resolution': 'resolution',
            'camera_source': 'camera_source',
            'camera_alias': 'camera_alias',
            'camera_autofocus': 'camera_autofocus',
            'camera_image_quality': 'camera_image_quality',
            'camerapi_focus_mode': 'camerapi_focus_mode',
            'camerapi_focus_distance_m': 'camerapi_focus_distance_m',
            'camerapi_awb_mode': 'camerapi_awb_mode',
            'camerapi_awb_red_gain': 'camerapi_awb_red_gain',
            'camerapi_awb_blue_gain': 'camerapi_awb_blue_gain',
            'camerapi_color_order': 'camerapi_color_order',
            'video': 'video_file',
            'tracking_mode': 'tracking_mode',
            'conf': 'confidence_threshold',
            'model_format': 'model_format',
            'model_path': 'model_path', 
            'class_map': 'class_map',
            'save': 'save_detections',
            'save_dir': 'save_directory',
            'detection_max_file_count': 'detection_max_file_count',
            'print': 'print_detections',
            'motion': 'enable_motion_detection',
            'min_motion_area': 'min_motion_area',
            'dilation': 'dilation_iterations',
            'brake': 'frame_delay',
            'dataset_delay': 'dataset_frame_delay',
            'web': 'enable_web',
            'web_host': 'web_host',
            'web_port': 'web_port',
        }
        
        for arg_key, config_key in arg_mapping.items():
            if hasattr(args, arg_key):
                value = getattr(args, arg_key)
                if value is not None:
                    self.config[config_key] = value
        
        # Handle SMS enable/disable flags
        if hasattr(args, 'sms') and args.sms:
            self.config['enable_sms'] = True
        elif hasattr(args, 'no_sms') and args.no_sms:
            self.config['enable_sms'] = False

        if hasattr(args, 'push') and args.push:
            self.config['enable_push'] = True
        elif hasattr(args, 'no_push') and args.no_push:
            self.config['enable_push'] = False

        if hasattr(args, 'no_camera_autofocus') and args.no_camera_autofocus:
            self.config['camera_autofocus'] = False
    
    def get(self, key: str, default=None) -> Any:
        """
        Get configuration value by key.
        
        Args:
            key: Configuration key
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        return self.config.get(key, default)
    
    def set(self, key: str, value: Any):
        """
        Set configuration value.
        
        Args:
            key: Configuration key
            value: Value to set
        """
        self.config[key] = value
        logger.debug("Set %s to %s", key, value)
    
    def get_camera_resolution(self) -> Tuple[int, int]:
        """
        Get camera resolution as (width, height) tuple.
        
        Returns:
            Tuple of (width, height)
        """
        from .detection import parse_resolution
        return parse_resolution(self.config['resolution'])
    
    def get_sms_config(self) -> Dict[str, Any]:
        """
        Get SMS configuration dictionary.
        
        Returns:
            Dictionary with SMS configuration
        """
        return {
            'enabled': self.config['enable_sms'],
            'api_key': self.config['lox24_api_key'],
            'phone_number': self.config['phone_number'],
            'sender_name': self.config['lox24_sender'],
            'delay_minutes': self.config['sms_delay_minutes'],
        }

    def get_push_config(self) -> Dict[str, Any]:
        """
        Get Pushover configuration dictionary.

        Returns:
            Dictionary with Pushover configuration
        """
        return {
            'enabled': self.config['enable_push'],
            'token': self.config['pushover_token'],
            'user': self.config['pushover_user'],
            'sender_name': self.config['pushover_sender'],
            'delay_minutes': self.config['push_delay_minutes'],
            'thumbnail': self.config['push_thumbnail'],
        }
    
    def get_web_config(self) -> Dict[str, Any]:
        """
        Get web server configuration dictionary.
        
        Returns:
            Dictionary with web configuration
        """
        protocol = 'https' if self.config['use_https'] else 'http'
        domain = self.config['domain_name']
        port = self.config['web_port']
        
        if port in (80, 443):
            public_url = f"{protocol}://{domain}"
        else:
            public_url = f"{protocol}://{domain}:{port}"
        
        return {
            'enabled': self.config['enable_web'],
            'host': self.config['web_host'],
            'port': self.config['web_port'],
            'public_url': public_url,
        }
    
    def validate(self) -> bool:
        """
        Validate configuration values.
        
        Returns:
            True if configuration is valid
            
        Raises:
            ValueError: If configuration is invalid
        """
        # Validate confidence threshold
        conf = self.config['confidence_threshold']
        if not (0.0 <= conf <= 1.0):
            raise ValueError(f"Confidence threshold must be between 0.0 and 1.0, got {conf}")
        
        # Validate resolution
        try:
            width, height = self.get_camera_resolution()
            if width <= 0 or height <= 0:
                raise ValueError("Resolution must have positive width and height")
        except Exception as e:
            raise ValueError(f"Invalid resolution format: {e}")

        camera_source = self._normalize_camera_source(self.config['camera_source'])
        if camera_source not in {'auto', 'usb', 'picamera2'}:
            raise ValueError(
                f"Camera source must be one of auto, usb, picamera2, got {self.config['camera_source']}"
            )
        self.config['camera_source'] = camera_source
        self.config['camera_alias'] = self._normalize_camera_alias(self.config.get('camera_alias', 'Camera'))

        quality_setting = str(self.config.get('camera_image_quality', 'max')).strip().lower()
        if quality_setting != 'max':
            try:
                quality_value = int(quality_setting)
            except ValueError as error:
                raise ValueError(
                    f"Camera image quality must be 'max' or an integer between 1 and 100, got {self.config.get('camera_image_quality')}"
                ) from error
            if not (1 <= quality_value <= 100):
                raise ValueError(
                    f"Camera image quality must be between 1 and 100 when numeric, got {quality_value}"
                )
            self.config['camera_image_quality'] = str(quality_value)
        else:
            self.config['camera_image_quality'] = 'max'

        tracking_mode = self._normalize_tracking_mode(self.config.get('tracking_mode'))
        if tracking_mode not in {'off', 'centroid', 'iou'}:
            raise ValueError(
                f"Tracking mode must be one of off, centroid, iou, got {self.config.get('tracking_mode')}"
            )
        self.config['tracking_mode'] = tracking_mode

        focus_mode = str(self.config.get('camerapi_focus_mode', 'continuous') or 'continuous').strip().lower()
        if focus_mode not in {'continuous', 'auto', 'manual', 'off'}:
            raise ValueError(
                f"Pi camera focus mode must be one of continuous, auto, manual, off, got {self.config.get('camerapi_focus_mode')}"
            )
        self.config['camerapi_focus_mode'] = focus_mode

        awb_mode = str(self.config.get('camerapi_awb_mode', 'auto') or 'auto').strip().lower()
        if awb_mode not in {'auto', 'incandescent', 'tungsten', 'fluorescent', 'indoor', 'daylight', 'cloudy', 'off', 'custom', 'manual'}:
            raise ValueError(
                f"Pi camera AWB mode is invalid: {self.config.get('camerapi_awb_mode')}"
            )
        self.config['camerapi_awb_mode'] = awb_mode

        for key in ('camerapi_focus_distance_m', 'camerapi_awb_red_gain', 'camerapi_awb_blue_gain'):
            value = float(self.config.get(key, 0.0))
            if value < 0.0:
                raise ValueError(f"{key} must be >= 0.0, got {value}")
            self.config[key] = value

        for key in ('web_color_scale_r', 'web_color_scale_g', 'web_color_scale_b'):
            value = float(self.config.get(key, 1.0))
            if value <= 0.0:
                raise ValueError(f"{key} must be > 0.0, got {value}")
            self.config[key] = value

        color_order = str(self.config.get('camerapi_color_order', 'bgr') or 'bgr').strip().lower()
        if color_order not in {'rgb', 'bgr'}:
            raise ValueError(f"camerapi_color_order must be 'rgb' or 'bgr', got {self.config.get('camerapi_color_order')}")
        self.config['camerapi_color_order'] = color_order

        model_format = self._normalize_model_format(self.config.get('model_format'))
        if model_format not in {'auto', 'onnx', 'ncnn'}:
            raise ValueError(
                f"Model format must be one of auto, onnx, ncnn, got {self.config.get('model_format')}"
            )
        self.config['model_format'] = model_format
        
        # Validate ports
        web_port = self.config['web_port']
        if not (1 <= web_port <= 65535):
            raise ValueError(f"Web port must be between 1 and 65535, got {web_port}")
        
        # Validate paths
        model_path = self.config['model_path']
        if not model_path:
            raise ValueError("Model path cannot be empty")

        retention_days = self.config['detection_retention_days']
        if retention_days < 0:
            raise ValueError(f"Detection retention days must be >= 0, got {retention_days}")

        max_file_count = int(self.config.get('detection_max_file_count', 250))
        if max_file_count < 0:
            raise ValueError(f"Detection max file count must be >= 0, got {max_file_count}")
        self.config['detection_max_file_count'] = max_file_count
        
        logger.info("Configuration validation passed")
        return True
    
    def print_summary(self):
        """Print a summary of the current configuration."""
        print("\n" + "="*60)
        print("VespAI Configuration Summary")
        print("="*60)
        
        print(f"Resolution: {self.config['resolution']}")
        print(f"Camera source: {self.config['camera_source']}")
        print(f"Camera alias: {self.config['camera_alias']}")
        print(f"Camera autofocus: {self.config['camera_autofocus']}")
        print(f"Camera image quality: {self.config['camera_image_quality']}")
        print(f"Pi camera focus mode: {self.config['camerapi_focus_mode']}")
        print(f"Pi camera focus distance: {self.config['camerapi_focus_distance_m']}m")
        print(f"Pi camera AWB mode: {self.config['camerapi_awb_mode']}")
        print(f"Pi camera AWB manual gains: R={self.config['camerapi_awb_red_gain']} B={self.config['camerapi_awb_blue_gain']}")
        print(f"Pi camera color order: {self.config['camerapi_color_order']}")
        print(
            "Web color scales (R,G,B): "
            f"{self.config['web_color_scale_r']}, "
            f"{self.config['web_color_scale_g']}, "
            f"{self.config['web_color_scale_b']}"
        )
        print(f"Tracking mode: {self.config['tracking_mode']}")
        print(f"Model format: {self.config['model_format']}")
        print(f"Confidence threshold: {self.config['confidence_threshold']}")
        print(f"Model path: {self.config['model_path']}")
        if self.config.get('class_map'):
            print(f"Class map: {self.config['class_map']}")
        print(f"Save detections: {self.config['save_detections']}")
        if self.config['save_detections']:
            print(f"Save directory: {self.config['save_directory']}")
            print(f"Detection retention: {self.config['detection_retention_days']} days")
            print(f"Detection max file count: {self.config['detection_max_file_count']}")
        
        print(f"Motion detection: {self.config['enable_motion_detection']}")
        print(f"Dataset frame delay: {self.config['dataset_frame_delay']}s")
        print(f"Web interface: {self.config['enable_web']}")
        if self.config['enable_web']:
            web_config = self.get_web_config()
            print(f"Web URL: {web_config['public_url']}")
        
        print(f"SMS alerts: {self.config['enable_sms']}")
        if self.config['enable_sms'] and self.config['lox24_api_key']:
            print(f"SMS delay: {self.config['sms_delay_minutes']} minutes")

        print(f"Pushover alerts: {self.config['enable_push']}")
        if self.config['enable_push'] and self.config['pushover_token']:
            print(f"Pushover delay: {self.config['push_delay_minutes']} minutes")
            print(f"Pushover thumbnail: {self.config['push_thumbnail']}")
        
        print("="*60 + "\n")


def create_config_from_args(args=None) -> VespAIConfig:
    """
    Create and configure VespAI configuration from command line arguments.
    
    Args:
        args: Command line arguments (None for sys.argv)
        
    Returns:
        VespAIConfig: Configured instance
    """
    config = VespAIConfig()
    config.parse_args(args)
    config.validate()
    return config