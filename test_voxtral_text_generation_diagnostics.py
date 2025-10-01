#!/usr/bin/env python3
"""
Comprehensive Voxtral Text Generation Diagnostics
Tests the Voxtral model directly with text input to isolate text generation issues
from speech-to-text transcription problems.
"""

import asyncio
import websockets
import json
import time
import re
from typing import List, Dict, Tuple

class VoxtralTextDiagnostics:
    def __init__(self):
        self.test_cases = [
            {
                "id": 1,
                "input": "Hello, how are you?",
                "expected_patterns": [
                    r"(hello|hi|hey).*(good|fine|well|great)",
                    r"(i am|i'm).*(good|fine|well|great|doing)",
                    r"(what about you|how about you|and you)",
                    r"(nice to meet|pleasure to meet|good to see)"
                ],
                "expected_type": "greeting_response",
                "description": "Simple greeting conversation"
            },
            {
                "id": 2,
                "input": "Tell me about Hyderabad which is located in India.",
                "expected_patterns": [
                    r"hyderabad.*(city|capital|located|telangana|andhra)",
                    r"(beautiful|historic|cultural|technology|IT)",
                    r"(india|telangana|andhra pradesh)",
                    r"(charminar|biriyani|culture|history|tech hub)"
                ],
                "expected_type": "informational_response",
                "description": "Factual information about a city"
            },
            {
                "id": 3,
                "input": "What are your hobbies?",
                "expected_patterns": [
                    r"(help|assist|learn|teach|share)",
                    r"(knowledge|information|conversation)",
                    r"(enjoy|like|love|interested)",
                    r"(people|users|questions|problems)"
                ],
                "expected_type": "personal_response",
                "description": "Personal question about AI capabilities"
            }
        ]
        
        self.quality_criteria = {
            "min_words": 5,
            "max_words": 50,
            "avoid_patterns": [
                r"^[!@#$%^&*(),.?\":{}|<>]+$",  # Only punctuation
                r"^\s*$",  # Only whitespace
                r"^(:|;|\*|\"|\(|\)|\[|\])+",  # Starts with formatting
                r"(let's play|you start|dialogue|conversation format)",  # Roleplay artifacts
            ],
            "require_patterns": [
                r"[a-zA-Z]",  # Must contain letters
                r"\w+",  # Must contain word characters
            ]
        }

    async def test_text_input_direct(self, text_input: str, test_id: int) -> Dict:
        """Test Voxtral model with direct text input (bypassing audio)"""
        
        print(f"\n🧪 TEST {test_id}: Direct Text Input")
        print(f"   Input: '{text_input}'")
        
        try:
            uri = "ws://localhost:8000/ws"
            async with websockets.connect(uri) as websocket:
                print("   ✅ Connected to server")
                
                # Send text-only message (simulating perfect transcription)
                message = {
                    "type": "text_input",  # Direct text input
                    "text": text_input,
                    "mode": "conversation",
                    "streaming": True,
                    "prompt": "",
                    "timestamp": int(time.time() * 1000)
                }
                
                print(f"   📤 Sending direct text input...")
                await websocket.send(json.dumps(message))
                
                # Collect response
                response_text = ""
                words_collected = []
                start_time = time.time()
                
                try:
                    while time.time() - start_time < 15:  # 15 second timeout
                        response = await asyncio.wait_for(websocket.recv(), timeout=3.0)
                        data = json.loads(response)
                        
                        if data.get('type') == 'streaming_chunk':
                            word = data.get('text', '').strip()
                            if word:
                                words_collected.append(word)
                                print(f"   📥 Word: '{word}'")
                        elif data.get('type') == 'response':
                            response_text = data.get('text', '').strip()
                            if response_text:
                                print(f"   📥 Complete response: '{response_text}'")
                                break
                        elif data.get('type') == 'error':
                            print(f"   ❌ Error: {data.get('message', '')}")
                            return {"success": False, "error": data.get('message', '')}
                            
                except asyncio.TimeoutError:
                    print("   ⏰ Response timeout")
                
                # Use collected words if no complete response
                final_response = response_text if response_text else ' '.join(words_collected)
                
                return {
                    "success": True,
                    "response": final_response,
                    "words": words_collected,
                    "response_time": time.time() - start_time
                }
                
        except Exception as e:
            print(f"   ❌ Connection error: {e}")
            return {"success": False, "error": str(e)}

    def assess_response_quality(self, response: str, test_case: Dict) -> Dict:
        """Assess the quality of a generated response"""
        
        assessment = {
            "response": response,
            "word_count": len(response.split()) if response else 0,
            "character_count": len(response),
            "quality_score": 0,
            "issues": [],
            "strengths": [],
            "pattern_matches": [],
            "overall_grade": "FAIL"
        }
        
        if not response or not response.strip():
            assessment["issues"].append("Empty or whitespace-only response")
            return assessment
        
        # Check word count
        word_count = assessment["word_count"]
        if word_count < self.quality_criteria["min_words"]:
            assessment["issues"].append(f"Too short ({word_count} words, min {self.quality_criteria['min_words']})")
        elif word_count > self.quality_criteria["max_words"]:
            assessment["issues"].append(f"Too long ({word_count} words, max {self.quality_criteria['max_words']})")
        else:
            assessment["strengths"].append(f"Appropriate length ({word_count} words)")
            assessment["quality_score"] += 20
        
        # Check for problematic patterns
        for pattern in self.quality_criteria["avoid_patterns"]:
            if re.search(pattern, response, re.IGNORECASE):
                assessment["issues"].append(f"Contains problematic pattern: {pattern}")
        
        # Check for required patterns
        for pattern in self.quality_criteria["require_patterns"]:
            if re.search(pattern, response):
                assessment["quality_score"] += 10
            else:
                assessment["issues"].append(f"Missing required pattern: {pattern}")
        
        # Check for expected content patterns
        pattern_score = 0
        for pattern in test_case["expected_patterns"]:
            if re.search(pattern, response, re.IGNORECASE):
                assessment["pattern_matches"].append(pattern)
                pattern_score += 1
        
        if pattern_score > 0:
            assessment["strengths"].append(f"Matches {pattern_score}/{len(test_case['expected_patterns'])} expected patterns")
            assessment["quality_score"] += (pattern_score / len(test_case["expected_patterns"])) * 40
        else:
            assessment["issues"].append("No expected content patterns found")
        
        # Check for conversational naturalness
        conversational_indicators = [
            r"\b(i|you|we|my|your|our)\b",  # Personal pronouns
            r"\b(am|are|is|was|were|have|has|do|does|did)\b",  # Common verbs
            r"\b(the|a|an|this|that|these|those)\b",  # Articles and demonstratives
        ]
        
        natural_score = 0
        for indicator in conversational_indicators:
            if re.search(indicator, response, re.IGNORECASE):
                natural_score += 1
        
        if natural_score >= 2:
            assessment["strengths"].append("Natural conversational language")
            assessment["quality_score"] += 20
        else:
            assessment["issues"].append("Lacks natural conversational language")
        
        # Determine overall grade
        if assessment["quality_score"] >= 80:
            assessment["overall_grade"] = "EXCELLENT"
        elif assessment["quality_score"] >= 60:
            assessment["overall_grade"] = "GOOD"
        elif assessment["quality_score"] >= 40:
            assessment["overall_grade"] = "FAIR"
        elif assessment["quality_score"] >= 20:
            assessment["overall_grade"] = "POOR"
        else:
            assessment["overall_grade"] = "FAIL"
        
        return assessment

    async def run_comprehensive_diagnostics(self):
        """Run all diagnostic tests and provide comprehensive analysis"""
        
        print("🚀 STARTING COMPREHENSIVE VOXTRAL TEXT GENERATION DIAGNOSTICS")
        print("=" * 80)
        print("🎯 OBJECTIVE: Isolate text generation issues from speech-to-text problems")
        print("📋 METHOD: Test Voxtral model with direct text input (bypass audio)")
        print("=" * 80)
        
        results = []
        
        for test_case in self.test_cases:
            print(f"\n{'='*60}")
            print(f"TEST CASE {test_case['id']}: {test_case['description'].upper()}")
            print(f"{'='*60}")
            
            # Run the test
            result = await self.test_text_input_direct(test_case["input"], test_case["id"])
            
            if result["success"]:
                # Assess response quality
                assessment = self.assess_response_quality(result["response"], test_case)
                
                print(f"\n📊 RESPONSE ANALYSIS:")
                print(f"   Response: '{assessment['response']}'")
                print(f"   Word Count: {assessment['word_count']}")
                print(f"   Quality Score: {assessment['quality_score']}/100")
                print(f"   Overall Grade: {assessment['overall_grade']}")
                
                if assessment["strengths"]:
                    print(f"   ✅ Strengths:")
                    for strength in assessment["strengths"]:
                        print(f"      • {strength}")
                
                if assessment["issues"]:
                    print(f"   ❌ Issues:")
                    for issue in assessment["issues"]:
                        print(f"      • {issue}")
                
                if assessment["pattern_matches"]:
                    print(f"   🎯 Pattern Matches:")
                    for pattern in assessment["pattern_matches"]:
                        print(f"      • {pattern}")
                
                result["assessment"] = assessment
            else:
                print(f"   ❌ TEST FAILED: {result.get('error', 'Unknown error')}")
            
            results.append({
                "test_case": test_case,
                "result": result
            })
            
            # Small delay between tests
            await asyncio.sleep(2)
        
        # Generate comprehensive analysis
        self.generate_final_analysis(results)
        
        return results

    def generate_final_analysis(self, results: List[Dict]):
        """Generate comprehensive analysis and recommendations"""
        
        print(f"\n{'='*80}")
        print("🎯 COMPREHENSIVE DIAGNOSTIC ANALYSIS")
        print(f"{'='*80}")
        
        successful_tests = [r for r in results if r["result"]["success"]]
        failed_tests = [r for r in results if not r["result"]["success"]]
        
        print(f"📊 TEST SUMMARY:")
        print(f"   Total Tests: {len(results)}")
        print(f"   Successful: {len(successful_tests)}")
        print(f"   Failed: {len(failed_tests)}")
        
        if not successful_tests:
            print(f"\n💥 CRITICAL ISSUE: ALL TESTS FAILED")
            print(f"   🔧 RECOMMENDATION: Check server connectivity and basic functionality")
            return
        
        # Analyze response quality
        quality_grades = []
        quality_scores = []
        
        for result in successful_tests:
            if "assessment" in result["result"]:
                assessment = result["result"]["assessment"]
                quality_grades.append(assessment["overall_grade"])
                quality_scores.append(assessment["quality_score"])
        
        if quality_scores:
            avg_score = sum(quality_scores) / len(quality_scores)
            print(f"\n📈 QUALITY ANALYSIS:")
            print(f"   Average Quality Score: {avg_score:.1f}/100")
            grade_counts = {g: quality_grades.count(g) for g in set(quality_grades)}
            print(f"   Grade Distribution: {grade_counts}")
        
        # Determine root cause
        excellent_responses = [g for g in quality_grades if g in ["EXCELLENT", "GOOD"]]
        poor_responses = [g for g in quality_grades if g in ["POOR", "FAIL"]]
        
        print(f"\n🔍 ROOT CAUSE ANALYSIS:")
        
        if len(excellent_responses) >= 2:
            print(f"   ✅ TEXT GENERATION IS WORKING WELL")
            print(f"   📊 {len(excellent_responses)}/{len(quality_grades)} responses are high quality")
            print(f"   🎯 CONCLUSION: Issue is likely in SPEECH-TO-TEXT pipeline")
            print(f"   🔧 RECOMMENDATION: Investigate audio transcription accuracy")
            print(f"      • Check microphone input quality")
            print(f"      • Verify audio preprocessing")
            print(f"      • Test with clear, slow speech")
            print(f"      • Check for background noise interference")
            
        elif len(poor_responses) >= 2:
            print(f"   ❌ TEXT GENERATION HAS ISSUES")
            print(f"   📊 {len(poor_responses)}/{len(quality_grades)} responses are poor quality")
            print(f"   🎯 CONCLUSION: Issue is in VOXTRAL MODEL configuration")
            print(f"   🔧 RECOMMENDATION: Fix text generation pipeline")
            print(f"      • Review conversation prompts")
            print(f"      • Adjust generation parameters")
            print(f"      • Check model temperature/sampling settings")
            print(f"      • Verify tokenization and decoding")
            
        else:
            print(f"   ⚠️  MIXED RESULTS - INCONSISTENT PERFORMANCE")
            print(f"   📊 Quality varies across test cases")
            print(f"   🎯 CONCLUSION: Issue may be in BOTH pipelines")
            print(f"   🔧 RECOMMENDATION: Investigate both areas")
            print(f"      • Test with more diverse inputs")
            print(f"      • Check for context-dependent issues")
            print(f"      • Monitor system resources and performance")
        
        # Specific recommendations per test case
        print(f"\n📋 TEST-SPECIFIC ANALYSIS:")
        for i, result in enumerate(successful_tests):
            test_case = result["test_case"]
            assessment = result["result"].get("assessment", {})
            grade = assessment.get("overall_grade", "UNKNOWN")
            
            print(f"   Test {test_case['id']} ({test_case['expected_type']}): {grade}")
            if grade in ["POOR", "FAIL"]:
                issues = assessment.get("issues", [])
                print(f"      Issues: {', '.join(issues[:2])}")  # Show first 2 issues
        
        print(f"\n{'='*80}")
        print("🎉 DIAGNOSTIC COMPLETE - Use analysis above to guide next steps")
        print(f"{'='*80}")

async def main():
    """Main diagnostic function"""
    diagnostics = VoxtralTextDiagnostics()
    await diagnostics.run_comprehensive_diagnostics()

if __name__ == "__main__":
    asyncio.run(main())
