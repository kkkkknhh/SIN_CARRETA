# coding=utf-8
"""
MINIMINIMOON Immutability Contract
==================================

Provides mechanisms to freeze and verify the integrity of the MINIMINIMOON integration.
This module ensures that critical components are not tampered with and results are reproducible.
"""
import base64
import hashlib
import hmac
import importlib
import inspect
import json
import logging
import pickle
import time
from pathlib import Path
from typing import Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("MINIMINIMOONImmutability")

# This should be in a secured environment variable or config in production
# For demo purposes only
HMAC_KEY = b"MINIMINIMOON_SECURE_KEY_2024"


class ImmutabilityContract:
    """
    Provides mechanisms to ensure the immutability and integrity of the MINIMINIMOON system.
    
    This class is responsible for:
    1. Creating cryptographic signatures of critical components
    2. Verifying system integrity against these signatures
    3. Detecting unauthorized modifications to core modules
    4. Ensuring reproducibility of results
    """
    
    def __init__(self):
        """Initialize the immutability contract"""
        self.integration_state = {}
        self.frozen_state = {}
        self.component_hashes = {}
        self.execution_history = []
        self.frozen = False
        self._initialize_contract()
    
    def _initialize_contract(self):
        """Initialize the immutability contract with the current system state"""
        logger.info("Initializing immutability contract")
        
        # List of critical modules to protect
        critical_modules = [
            "embedding_model",
            "responsibility_detector", 
            "contradiction_detector",
            "monetary_detector",
            "feasibility_scorer",
            "teoria_cambio",
            "dag_validation",
            "decalogo_loader",
            "plan_sanitizer",
            "plan_processor",
            "document_segmenter",
            "document_embedding_mapper",
            "spacy_loader",
            "causal_pattern_detector"
        ]
        
        # Create initial hashes of all critical modules
        for module_name in critical_modules:
            try:
                module_hash = self._hash_module(module_name)
                self.component_hashes[module_name] = module_hash
                logger.debug(f"Registered module hash for {module_name}")
            except ImportError:
                logger.warning(f"Could not import module {module_name} for hashing")
            except Exception as e:
                logger.error(f"Error hashing module {module_name}: {e}")
        
        # Store the contract creation time
        self.integration_state["creation_time"] = time.time()
        self.integration_state["component_hashes"] = self.component_hashes.copy()
        self.integration_state["system_info"] = self._get_system_info()
        
        # Generate contract signature
        self.integration_state["signature"] = self._generate_signature(self.integration_state)
        
        logger.info(f"Immutability contract initialized with {len(critical_modules)} modules")
    
    def _hash_module(self, module_name: str) -> Dict[str, str]:
        """
        Calculate cryptographic hashes of a module's functions and methods
        
        Args:
            module_name: Name of the module to hash
            
        Returns:
            Dictionary mapping function names to their hashes
        """
        try:
            module = importlib.import_module(module_name)
            module_elements = {}
            
            # Hash module source code if available
            if hasattr(module, "__file__") and module.__file__:
                try:
                    with open(module.__file__, 'rb') as f:
                        module_source = f.read()
                        module_elements["__source__"] = hashlib.sha256(module_source).hexdigest()
                except Exception as e:
                    logger.warning(f"Could not hash source of {module_name}: {e}")
            
            # Hash functions and methods
            for name, obj in inspect.getmembers(module):
                if name.startswith("_") and name != "__init__":
                    continue
                    
                if inspect.isfunction(obj) or inspect.ismethod(obj):
                    try:
                        # Get function source if available
                        source = inspect.getsource(obj)
                        module_elements[name] = hashlib.sha256(source.encode()).hexdigest()
                    except Exception:
                        # For built-in functions without source
                        module_elements[name] = str(obj)
                elif inspect.isclass(obj):
                    class_elements = {}
                    for method_name, method in inspect.getmembers(obj, inspect.isfunction):
                        if method_name.startswith("_") and method_name != "__init__":
                            continue
                        try:
                            source = inspect.getsource(method)
                            class_elements[method_name] = hashlib.sha256(source.encode()).hexdigest()
                        except Exception:
                            class_elements[method_name] = str(method)
                    
                    if class_elements:
                        module_elements[name] = class_elements
            
            return module_elements
            
        except ImportError:
            logger.warning(f"Module {module_name} not found")
            return {}
        except Exception as e:
            logger.error(f"Error hashing module {module_name}: {e}")
            return {"error": str(e)}
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Gather basic system information for reproducibility"""
        info = {
            "python_version": sys.version,
            "platform": sys.platform,
            "modules": {}
        }
        
        # Collect version information of key modules
        for module_name in ["spacy", "numpy", "sentence_transformers", "torch", "networkx"]:
            try:
                module = importlib.import_module(module_name)
                info["modules"][module_name] = getattr(module, "__version__", "unknown")
            except ImportError:
                info["modules"][module_name] = "not installed"
        
        return info
    
    def _generate_signature(self, data: Dict[str, Any]) -> str:
        """
        Generate a cryptographic signature for the given data
        
        Args:
            data: Data to sign
            
        Returns:
            Base64-encoded HMAC signature
        """
        # Serialize the data
        serialized = pickle.dumps(data)
        
        # Generate HMAC signature
        signature = hmac.new(HMAC_KEY, serialized, hashlib.sha256).digest()
        
        # Return base64 encoded signature
        return base64.b64encode(signature).decode('utf-8')
    
    def _verify_signature(self, data: Dict[str, Any], signature: str) -> bool:
        """
        Verify a cryptographic signature for the given data
        
        Args:
            data: Data to verify
            signature: Expected signature
            
        Returns:
            True if signature is valid, False otherwise
        """
        # Serialize the data
        serialized = pickle.dumps(data)
        
        # Generate HMAC signature
        expected_signature = hmac.new(HMAC_KEY, serialized, hashlib.sha256).digest()
        expected_b64 = base64.b64encode(expected_signature).decode('utf-8')
        
        # Compare signatures
        return hmac.compare_digest(expected_b64, signature)
    
    def verify_components(self, verification_level: str = "normal") -> Dict[str, Any]:
        """
        Verify the integrity of system components
        
        Args:
            verification_level: Level of verification (minimal, normal, strict)
            
        Returns:
            Dictionary with verification results
        """
        if not self.frozen:
            logger.warning("System state not frozen yet, verification is less reliable")
        
        verification_result = {
            "verification_time": time.time(),
            "verification_level": verification_level,
            "verified_components": 0,
            "modified_components": [],
            "missing_components": [],
            "integrity_status": "not_verified"
        }
        
        # Get modules to verify based on verification level
        modules_to_verify = set(self.component_hashes.keys())
        if verification_level == "minimal":
            # Only verify core processing modules
            core_modules = {"embedding_model", "responsibility_detector", "teoria_cambio"}
            modules_to_verify = modules_to_verify.intersection(core_modules)
        elif verification_level == "strict":
            # Add utility and support modules
            additional_modules = {
                "utils", "text_processor", "pdm_nlp_modules", 
                "pdm_nli_policy_modules", "pdm_contra_main"
            }
            modules_to_verify.update(additional_modules)
        
        # Verify each module
        for module_name in modules_to_verify:
            try:
                current_hash = self._hash_module(module_name)
                original_hash = self.component_hashes.get(module_name, {})
                
                if not original_hash:
                    verification_result["missing_components"].append(module_name)
                    continue
                
                # Compare function hashes
                modified_functions = []
                for func_name, func_hash in original_hash.items():
                    if func_name == "__source__":
                        # Source code comparison
                        if func_hash != current_hash.get("__source__"):
                            modified_functions.append(f"{module_name} (source)")
                    elif isinstance(func_hash, dict):
                        # Class methods
                        for method_name, method_hash in func_hash.items():
                            current_class_methods = current_hash.get(func_name, {})
                            if method_hash != current_class_methods.get(method_name):
                                modified_functions.append(f"{module_name}.{func_name}.{method_name}")
                    elif func_hash != current_hash.get(func_name):
                        # Regular functions
                        modified_functions.append(f"{module_name}.{func_name}")
                
                if modified_functions:
                    verification_result["modified_components"].append({
                        "module": module_name,
                        "modified_functions": modified_functions
                    })
                else:
                    verification_result["verified_components"] += 1
                
            except Exception as e:
                logger.error(f"Error verifying module {module_name}: {e}")
                verification_result["missing_components"].append(module_name)
        
        # Determine integrity status
        if not verification_result["modified_components"] and not verification_result["missing_components"]:
            verification_result["integrity_status"] = "verified"
        elif not verification_result["modified_components"] and verification_result["missing_components"]:
            verification_result["integrity_status"] = "incomplete"
        else:
            verification_result["integrity_status"] = "modified"
        
        logger.info(f"Component verification completed: {verification_result['integrity_status']}")
        return verification_result
    
    def freeze_integration(self) -> Dict[str, Any]:
        """
        Freeze the current integration state to establish an immutability baseline.
        
        Returns:
            Dictionary with freeze status information
        """
        if self.frozen:
            return {
                "status": "already_frozen", 
                "freeze_time": self.frozen_state.get("freeze_time"),
                "message": "Integration is already frozen"
            }
        
        logger.info("Freezing MINIMINIMOON integration state")
        
        # Collect current component state
        component_state = {}
        for module_name in self.component_hashes.keys():
            component_state[module_name] = self._hash_module(module_name)
        
        # Create freeze record
        freeze_record = {
            "freeze_time": time.time(),
            "component_state": component_state,
            "system_info": self._get_system_info(),
            "freeze_version": "1.0"
        }
        
        # Generate cryptographic signature
        freeze_record["signature"] = self._generate_signature(freeze_record)
        
        # Store freeze state
        self.frozen_state = freeze_record
        self.frozen = True
        
        # Save freeze record to file
        try:
            freeze_path = Path("/Users/recovered/MINIMINIMOON/integration_freeze.json")
            with open(freeze_path, 'w', encoding='utf-8') as f:
                # Remove complex objects that can't be JSON serialized
                save_record = {k: v for k, v in freeze_record.items() if k != "system_info"}
                save_record["system_info"] = {
                    "python_version": freeze_record["system_info"]["python_version"],
                    "platform": freeze_record["system_info"]["platform"],
                    "modules": freeze_record["system_info"]["modules"]
                }
                json.dump(save_record, f, indent=2)
            logger.info(f"Freeze record saved to {freeze_path}")
        except Exception as e:
            logger.error(f"Error saving freeze record: {e}")
        
        logger.info("Integration state successfully frozen")
        return {
            "status": "success",
            "freeze_time": freeze_record["freeze_time"],
            "modules_frozen": len(component_state),
            "freeze_file": str(freeze_path) if 'freeze_path' in locals() else None
        }
    
    def verify_integration(self) -> Dict[str, Any]:
        """
        Verify that the integration hasn't been tampered with.
        
        Returns:
            Dictionary with verification results
        """
        if not self.frozen:
            return {
                "status": "not_frozen",
                "message": "Integration not frozen, cannot verify"
            }
        
        logger.info("Verifying integration against frozen state")
        
        # First verify the signature of the frozen state
        frozen_data = {k: v for k, v in self.frozen_state.items() if k != "signature"}
        expected_signature = self.frozen_state["signature"]
        
        if not self._verify_signature(frozen_data, expected_signature):
            return {
                "status": "tampered_record",
                "message": "Freeze record has been tampered with",
                "verification_time": time.time()
            }
        
        # Now verify each component against its frozen state
        verification_result = {
            "status": "verifying",
            "verification_time": time.time(),
            "verified_components": 0,
            "modified_components": [],
            "missing_components": []
        }
        
        frozen_components = self.frozen_state.get("component_state", {})
        for module_name, frozen_hash in frozen_components.items():
            try:
                current_hash = self._hash_module(module_name)
                
                # Compare module hashes
                modified_functions = []
                for func_name, func_hash in frozen_hash.items():
                    if func_name == "__source__":
                        # Source code comparison
                        if func_hash != current_hash.get("__source__"):
                            modified_functions.append(f"{module_name} (source)")
                    elif isinstance(func_hash, dict):
                        # Class methods
                        for method_name, method_hash in func_hash.items():
                            current_class_methods = current_hash.get(func_name, {})
                            if method_hash != current_class_methods.get(method_name):
                                modified_functions.append(f"{module_name}.{func_name}.{method_name}")
                    elif func_hash != current_hash.get(func_name):
                        # Regular functions
                        modified_functions.append(f"{module_name}.{func_name}")
                
                if modified_functions:
                    verification_result["modified_components"].append({
                        "module": module_name,
                        "modified_functions": modified_functions
                    })
                else:
                    verification_result["verified_components"] += 1
                
            except ImportError:
                verification_result["missing_components"].append(module_name)
            except Exception as e:
                logger.error(f"Error verifying module {module_name}: {e}")
                verification_result["missing_components"].append(f"{module_name} (error)")
        
        # Determine overall status
        if not verification_result["modified_components"] and not verification_result["missing_components"]:
            verification_result["status"] = "verified"
            verification_result["message"] = "Integration verified successfully"
        elif not verification_result["modified_components"]:
            verification_result["status"] = "incomplete"
            verification_result["message"] = f"Integration missing {len(verification_result['missing_components'])} components"
        else:
            verification_result["status"] = "modified"
            verification_result["message"] = f"Integration has {len(verification_result['modified_components'])} modified components"
        
        logger.info(f"Integration verification result: {verification_result['status']}")
        return verification_result
    
    def register_process_execution(self, results: Dict[str, Any]) -> None:
        """
        Register the execution of a process for audit trail
        
        Args:
            results: Processing results to register
        """
        execution_record = {
            "timestamp": time.time(),
            "result_hash": self.generate_result_hash(results),
            "components_used": list(self.component_hashes.keys()),
            "plan_name": results.get("plan_name", "unknown")
        }
        
        self.execution_history.append(execution_record)
        logger.debug(f"Registered execution: {execution_record['result_hash']}")
        
        # Limit history size
        if len(self.execution_history) > 100:
            self.execution_history = self.execution_history[-100:]
    
    def generate_result_hash(self, results: Dict[str, Any]) -> str:
        """
        Generate a deterministic hash for processing results
        
        Args:
            results: Results to hash
            
        Returns:
            Hash of the results
        """
        # Remove non-deterministic elements
        clean_results = {k: v for k, v in results.items() 
                        if k not in ["execution_summary", "immutability_hash"]}
        
        # Sort dictionaries to ensure deterministic order
        serialized = json.dumps(clean_results, sort_keys=True).encode()
        
        # Generate hash
        result_hash = hashlib.sha256(serialized).hexdigest()
        return result_hash
    
    def verify_result_reproducibility(self, result_hash: str) -> Dict[str, Any]:
        """
        Verify if a result hash matches any previously recorded execution
        
        Args:
            result_hash: Hash to verify
            
        Returns:
            Dictionary with verification results
        """
        for execution in self.execution_history:
            if execution["result_hash"] == result_hash:
                return {
                    "reproducible": True,
                    "original_timestamp": execution["timestamp"],
                    "plan_name": execution["plan_name"],
                    "verification_time": time.time()
                }
        
        return {
            "reproducible": False,
            "verification_time": time.time(),
            "message": "No matching execution found in history"
        }


if __name__ == "__main__":
    import sys
    
    print("MINIMINIMOON Immutability Contract Tool")
    print("======================================")
    
    contract = ImmutabilityContract()
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "verify":
            level = sys.argv[2] if len(sys.argv) > 2 else "normal"
            print(f"Verifying components with level: {level}")
            result = contract.verify_components(level)
            print(f"Verification status: {result['integrity_status']}")
            print(f"Verified components: {result['verified_components']}")
            
            if result['modified_components']:
                print("\nModified components:")
                for comp in result['modified_components']:
                    print(f"- {comp['module']}")
            
            if result['missing_components']:
                print("\nMissing components:")
                for comp in result['missing_components']:
                    print(f"- {comp}")
        
        elif command == "freeze":
            print("Freezing integration state...")
            result = contract.freeze_integration()
            print(f"Freeze status: {result['status']}")
            if result['status'] == 'success':
                print(f"Freeze file: {result['freeze_file']}")
        
        else:
            print(f"Unknown command: {command}")
            print("Available commands: verify, freeze")
    
    else:
        print("\nAvailable commands:")
        print("  python miniminimoon_immutability.py verify [minimal|normal|strict]")
        print("  python miniminimoon_immutability.py freeze")
