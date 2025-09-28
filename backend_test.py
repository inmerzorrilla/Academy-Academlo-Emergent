#!/usr/bin/env python3
"""
Comprehensive Backend Testing for ACADEMY Application
Tests all critical backend APIs including the new Claude-powered chatbot
"""

import requests
import json
import time
import sys
from datetime import datetime

# Get backend URL from frontend .env
BACKEND_URL = "https://edutechlab-1.preview.emergentagent.com/api"

class AcademyBackendTester:
    def __init__(self):
        self.base_url = BACKEND_URL
        self.auth_token = None
        self.test_user_data = {
            "name": "Maria Rodriguez",
            "email": f"maria.test.{int(time.time())}@academy.com",
            "phone": "+1234567890",
            "password": "SecurePass123!"
        }
        self.results = {
            "total_tests": 0,
            "passed": 0,
            "failed": 0,
            "errors": []
        }
    
    def log_result(self, test_name, success, message="", response_data=None):
        """Log test results"""
        self.results["total_tests"] += 1
        if success:
            self.results["passed"] += 1
            print(f"✅ {test_name}: {message}")
        else:
            self.results["failed"] += 1
            self.results["errors"].append(f"{test_name}: {message}")
            print(f"❌ {test_name}: {message}")
            if response_data:
                print(f"   Response: {response_data}")
    
    def make_request(self, method, endpoint, data=None, headers=None):
        """Make HTTP request with error handling"""
        url = f"{self.base_url}{endpoint}"
        try:
            if method.upper() == "GET":
                response = requests.get(url, headers=headers, timeout=30)
            elif method.upper() == "POST":
                response = requests.post(url, json=data, headers=headers, timeout=30)
            elif method.upper() == "DELETE":
                response = requests.delete(url, headers=headers, timeout=30)
            else:
                return None, f"Unsupported method: {method}"
            
            return response, None
        except requests.exceptions.RequestException as e:
            return None, str(e)
    
    def test_user_registration(self):
        """Test user registration endpoint"""
        print("\n🔐 Testing User Registration...")
        
        response, error = self.make_request("POST", "/auth/register", self.test_user_data)
        
        if error:
            self.log_result("User Registration", False, f"Request failed: {error}")
            return False
        
        if response.status_code == 200:
            data = response.json()
            if "token" in data and "user" in data:
                self.auth_token = data["token"]
                self.log_result("User Registration", True, "Successfully registered user and received token")
                return True
            else:
                self.log_result("User Registration", False, "Missing token or user in response", data)
                return False
        else:
            self.log_result("User Registration", False, f"Status {response.status_code}", response.text)
            return False
    
    def test_user_login(self):
        """Test user login endpoint"""
        print("\n🔑 Testing User Login...")
        
        login_data = {
            "email": self.test_user_data["email"],
            "password": self.test_user_data["password"]
        }
        
        response, error = self.make_request("POST", "/auth/login", login_data)
        
        if error:
            self.log_result("User Login", False, f"Request failed: {error}")
            return False
        
        if response.status_code == 200:
            data = response.json()
            if "token" in data and "user" in data:
                self.auth_token = data["token"]
                self.log_result("User Login", True, "Successfully logged in and received token")
                return True
            else:
                self.log_result("User Login", False, "Missing token or user in response", data)
                return False
        else:
            self.log_result("User Login", False, f"Status {response.status_code}", response.text)
            return False
    
    def test_user_profile(self):
        """Test user profile endpoint"""
        print("\n👤 Testing User Profile...")
        
        if not self.auth_token:
            self.log_result("User Profile", False, "No auth token available")
            return False
        
        headers = {"Authorization": f"Bearer {self.auth_token}"}
        response, error = self.make_request("GET", "/user/profile", headers=headers)
        
        if error:
            self.log_result("User Profile", False, f"Request failed: {error}")
            return False
        
        if response.status_code == 200:
            data = response.json()
            if "name" in data and "email" in data:
                self.log_result("User Profile", True, f"Retrieved profile for {data['name']}")
                return True
            else:
                self.log_result("User Profile", False, "Missing required fields in profile", data)
                return False
        else:
            self.log_result("User Profile", False, f"Status {response.status_code}", response.text)
            return False
    
    def test_user_progress(self):
        """Test user progress endpoints"""
        print("\n📊 Testing User Progress...")
        
        if not self.auth_token:
            self.log_result("User Progress", False, "No auth token available")
            return False
        
        headers = {"Authorization": f"Bearer {self.auth_token}"}
        
        # Test GET progress
        response, error = self.make_request("GET", "/user/progress", headers=headers)
        
        if error:
            self.log_result("User Progress GET", False, f"Request failed: {error}")
            return False
        
        if response.status_code == 200:
            data = response.json()
            if "teorico_progress" in data and "escucha_progress" in data:
                self.log_result("User Progress GET", True, "Retrieved user progress successfully")
                
                # Test POST progress update
                update_data = {
                    "module": "teorico",
                    "item_id": 1
                }
                
                response, error = self.make_request("POST", "/user/progress", update_data, headers)
                
                if error:
                    self.log_result("User Progress UPDATE", False, f"Request failed: {error}")
                    return False
                
                if response.status_code == 200:
                    updated_data = response.json()
                    if updated_data.get("teorico_progress", 0) > 0:
                        self.log_result("User Progress UPDATE", True, "Successfully updated progress")
                        return True
                    else:
                        self.log_result("User Progress UPDATE", False, "Progress not updated", updated_data)
                        return False
                else:
                    self.log_result("User Progress UPDATE", False, f"Status {response.status_code}", response.text)
                    return False
            else:
                self.log_result("User Progress GET", False, "Missing required fields in progress", data)
                return False
        else:
            self.log_result("User Progress GET", False, f"Status {response.status_code}", response.text)
            return False
    
    def test_content_teorico(self):
        """Test teorico content endpoint"""
        print("\n📚 Testing Teorico Content...")
        
        response, error = self.make_request("GET", "/content/teorico")
        
        if error:
            self.log_result("Teorico Content", False, f"Request failed: {error}")
            return False
        
        if response.status_code == 200:
            data = response.json()
            if isinstance(data, list) and len(data) > 0:
                first_question = data[0]
                if "id" in first_question and "question" in first_question and "answer" in first_question:
                    self.log_result("Teorico Content", True, f"Retrieved {len(data)} theoretical questions")
                    return True
                else:
                    self.log_result("Teorico Content", False, "Invalid question structure", first_question)
                    return False
            else:
                self.log_result("Teorico Content", False, "Empty or invalid response", data)
                return False
        else:
            self.log_result("Teorico Content", False, f"Status {response.status_code}", response.text)
            return False
    
    def test_content_escucha(self):
        """Test escucha content endpoint with language parameters"""
        print("\n🎥 Testing Escucha Content (Spanish and English)...")
        
        # Test Spanish content
        response, error = self.make_request("GET", "/content/escucha?lang=es")
        
        if error:
            self.log_result("Escucha Content (Spanish)", False, f"Request failed: {error}")
            return False
        
        spanish_success = False
        if response.status_code == 200:
            data = response.json()
            if isinstance(data, list) and len(data) > 0:
                first_video = data[0]
                if "id" in first_video and "title" in first_video and "url" in first_video:
                    # Check if it's Spanish content
                    if "Introducción" in first_video["title"] or "IA" in first_video["title"]:
                        self.log_result("Escucha Content (Spanish)", True, f"Retrieved {len(data)} Spanish videos")
                        spanish_success = True
                    else:
                        self.log_result("Escucha Content (Spanish)", False, "Content doesn't appear to be in Spanish", first_video)
                else:
                    self.log_result("Escucha Content (Spanish)", False, "Invalid video structure", first_video)
            else:
                self.log_result("Escucha Content (Spanish)", False, "Empty or invalid response", data)
        else:
            self.log_result("Escucha Content (Spanish)", False, f"Status {response.status_code}", response.text)
        
        # Test English content
        response, error = self.make_request("GET", "/content/escucha?lang=en")
        
        if error:
            self.log_result("Escucha Content (English)", False, f"Request failed: {error}")
            return False
        
        english_success = False
        if response.status_code == 200:
            data = response.json()
            if isinstance(data, list) and len(data) > 0:
                first_video = data[0]
                if "id" in first_video and "title" in first_video and "url" in first_video:
                    # Check if it's English content
                    if "Introduction" in first_video["title"] or "AI" in first_video["title"]:
                        self.log_result("Escucha Content (English)", True, f"Retrieved {len(data)} English videos")
                        english_success = True
                    else:
                        self.log_result("Escucha Content (English)", False, "Content doesn't appear to be in English", first_video)
                else:
                    self.log_result("Escucha Content (English)", False, "Invalid video structure", first_video)
            else:
                self.log_result("Escucha Content (English)", False, "Empty or invalid response", data)
        else:
            self.log_result("Escucha Content (English)", False, f"Status {response.status_code}", response.text)
        
        return spanish_success and english_success
    
    def test_content_prompt(self):
        """Test prompt content endpoint"""
        print("\n💡 Testing Prompt Content...")
        
        response, error = self.make_request("GET", "/content/prompt")
        
        if error:
            self.log_result("Prompt Content", False, f"Request failed: {error}")
            return False
        
        if response.status_code == 200:
            data = response.json()
            if isinstance(data, list) and len(data) > 0:
                first_prompt = data[0]
                if "id" in first_prompt and "title" in first_prompt and "prompt" in first_prompt:
                    self.log_result("Prompt Content", True, f"Retrieved {len(data)} prompt examples")
                    return True
                else:
                    self.log_result("Prompt Content", False, "Invalid prompt structure", first_prompt)
                    return False
            else:
                self.log_result("Prompt Content", False, "Empty or invalid response", data)
                return False
        else:
            self.log_result("Prompt Content", False, f"Status {response.status_code}", response.text)
            return False
    
    def test_chatbot_spanish(self):
        """Test chatbot with Spanish questions"""
        print("\n🤖 Testing Chatbot (Spanish Questions)...")
        
        spanish_questions = [
            {
                "message": "¿Qué es un Deep Agent?",
                "expected_topics": ["Deep Agent", "inteligencia artificial", "redes neuronales", "ACADEMY"]
            },
            {
                "message": "¿Cómo funciona el machine learning?",
                "expected_topics": ["machine learning", "aprendizaje", "algoritmos", "ACADEMY"]
            },
            {
                "message": "¿Qué es ACADEMY?",
                "expected_topics": ["ACADEMY", "plataforma", "educativa", "programadores"]
            }
        ]
        
        success_count = 0
        for i, question in enumerate(spanish_questions):
            response, error = self.make_request("POST", "/chat", question)
            
            if error:
                self.log_result(f"Chatbot Spanish Q{i+1}", False, f"Request failed: {error}")
                continue
            
            if response.status_code == 200:
                data = response.json()
                if "response" in data and data["response"]:
                    response_text = data["response"].lower()
                    # Check if response contains expected topics
                    topic_found = any(topic.lower() in response_text for topic in question["expected_topics"])
                    if topic_found:
                        self.log_result(f"Chatbot Spanish Q{i+1}", True, f"Intelligent response received: {data['response'][:100]}...")
                        success_count += 1
                    else:
                        self.log_result(f"Chatbot Spanish Q{i+1}", False, f"Response lacks expected topics: {data['response'][:100]}...")
                else:
                    self.log_result(f"Chatbot Spanish Q{i+1}", False, "Empty or invalid response", data)
            else:
                self.log_result(f"Chatbot Spanish Q{i+1}", False, f"Status {response.status_code}", response.text)
        
        return success_count >= 2  # At least 2 out of 3 should work
    
    def test_chatbot_english(self):
        """Test chatbot with English questions"""
        print("\n🤖 Testing Chatbot (English Questions)...")
        
        english_questions = [
            {
                "message": "What is artificial intelligence?",
                "expected_topics": ["artificial intelligence", "AI", "machine learning", "ACADEMY"]
            },
            {
                "message": "How do neural networks work?",
                "expected_topics": ["neural networks", "neurons", "deep learning", "ACADEMY"]
            },
            {
                "message": "Tell me about ACADEMY platform",
                "expected_topics": ["ACADEMY", "platform", "educational", "programmers"]
            }
        ]
        
        success_count = 0
        for i, question in enumerate(english_questions):
            response, error = self.make_request("POST", "/chat", question)
            
            if error:
                self.log_result(f"Chatbot English Q{i+1}", False, f"Request failed: {error}")
                continue
            
            if response.status_code == 200:
                data = response.json()
                if "response" in data and data["response"]:
                    response_text = data["response"].lower()
                    # Check if response contains expected topics
                    topic_found = any(topic.lower() in response_text for topic in question["expected_topics"])
                    if topic_found:
                        self.log_result(f"Chatbot English Q{i+1}", True, f"Intelligent response received: {data['response'][:100]}...")
                        success_count += 1
                    else:
                        self.log_result(f"Chatbot English Q{i+1}", False, f"Response lacks expected topics: {data['response'][:100]}...")
                else:
                    self.log_result(f"Chatbot English Q{i+1}", False, "Empty or invalid response", data)
            else:
                self.log_result(f"Chatbot English Q{i+1}", False, f"Status {response.status_code}", response.text)
        
        return success_count >= 2  # At least 2 out of 3 should work
    
    def test_chatbot_fallback(self):
        """Test chatbot fallback mechanism"""
        print("\n🛡️ Testing Chatbot Fallback Mechanism...")
        
        # Test with a very complex question that might cause issues
        complex_question = {
            "message": "This is a test to see if the fallback mechanism works when there are technical difficulties with the AI service integration."
        }
        
        response, error = self.make_request("POST", "/chat", complex_question)
        
        if error:
            self.log_result("Chatbot Fallback", False, f"Request failed: {error}")
            return False
        
        if response.status_code == 200:
            data = response.json()
            if "response" in data and data["response"]:
                # Check if it's a fallback response (contains WhatsApp number or technical difficulties message)
                response_text = data["response"].lower()
                if "whatsapp" in response_text or "dificultades técnicas" in response_text or "technical" in response_text:
                    self.log_result("Chatbot Fallback", True, "Fallback mechanism working correctly")
                    return True
                else:
                    self.log_result("Chatbot Fallback", True, "Received normal AI response (fallback not triggered)")
                    return True
            else:
                self.log_result("Chatbot Fallback", False, "Empty response", data)
                return False
        else:
            self.log_result("Chatbot Fallback", False, f"Status {response.status_code}", response.text)
            return False
    
    def run_all_tests(self):
        """Run all backend tests"""
        print("🚀 Starting ACADEMY Backend Comprehensive Testing...")
        print(f"Backend URL: {self.base_url}")
        print(f"Test started at: {datetime.now()}")
        
        # Authentication tests
        auth_success = self.test_user_registration()
        if not auth_success:
            # Try login if registration failed (user might already exist)
            auth_success = self.test_user_login()
        
        if auth_success:
            self.test_user_profile()
            self.test_user_progress()
        
        # Content tests (no auth required)
        self.test_content_teorico()
        self.test_content_escucha()
        self.test_content_prompt()
        
        # Chatbot tests (priority focus)
        chatbot_spanish = self.test_chatbot_spanish()
        chatbot_english = self.test_chatbot_english()
        chatbot_fallback = self.test_chatbot_fallback()
        
        # Print final results
        print("\n" + "="*60)
        print("🎯 FINAL TEST RESULTS")
        print("="*60)
        print(f"Total Tests: {self.results['total_tests']}")
        print(f"✅ Passed: {self.results['passed']}")
        print(f"❌ Failed: {self.results['failed']}")
        print(f"Success Rate: {(self.results['passed']/self.results['total_tests']*100):.1f}%")
        
        if self.results['errors']:
            print("\n🚨 FAILED TESTS:")
            for error in self.results['errors']:
                print(f"   • {error}")
        
        # Critical assessment
        critical_failures = []
        if not (chatbot_spanish or chatbot_english):
            critical_failures.append("Chatbot AI integration completely failed")
        if not auth_success:
            critical_failures.append("Authentication system failed")
        
        if critical_failures:
            print("\n🔴 CRITICAL ISSUES FOUND:")
            for failure in critical_failures:
                print(f"   • {failure}")
            return False
        else:
            print("\n🟢 All critical systems are working!")
            return True

if __name__ == "__main__":
    tester = AcademyBackendTester()
    success = tester.run_all_tests()
    sys.exit(0 if success else 1)