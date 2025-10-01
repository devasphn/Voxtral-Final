"""
Voice Profiles for Kokoro TTS
Provides detailed information about available voices with quality ratings and characteristics
"""

from typing import Dict, List, Any
from dataclasses import dataclass

@dataclass
class VoiceProfile:
    """Voice profile with detailed characteristics"""
    voice_id: str
    name: str
    gender: str
    language: str
    accent: str
    quality_grade: str
    description: str
    recommended_for: List[str]
    sample_rate: int = 24000

class VoiceProfileManager:
    """Manages voice profiles and recommendations"""
    
    def __init__(self):
        self.profiles = self._initialize_profiles()
    
    def _initialize_profiles(self) -> Dict[str, VoiceProfile]:
        """Initialize all available voice profiles"""
        profiles = {}
        
        # Hindi Female Voices (Indian Accent)
        profiles["hf_alpha"] = VoiceProfile(
            voice_id="hf_alpha",
            name="Priya (Hindi Female Alpha)",
            gender="female",
            language="hindi",
            accent="indian",
            quality_grade="C",
            description="Clear Hindi female voice with natural Indian accent. Excellent for English with Indian pronunciation.",
            recommended_for=["indian_accent", "hindi_english", "conversational", "clear_speech"]
        )
        
        profiles["hf_beta"] = VoiceProfile(
            voice_id="hf_beta",
            name="Ananya (Hindi Female Beta)",
            gender="female",
            language="hindi",
            accent="indian",
            quality_grade="C",
            description="Alternative Hindi female voice with warm Indian accent. Good for varied speech patterns.",
            recommended_for=["indian_accent", "warm_tone", "alternative_voice", "expressive"]
        )
        
        # High-Quality English Female Voices
        profiles["af_bella"] = VoiceProfile(
            voice_id="af_bella",
            name="Bella (American Female)",
            gender="female",
            language="english",
            accent="american",
            quality_grade="A-",
            description="Warm and friendly American female voice with excellent clarity. Premium quality.",
            recommended_for=["high_quality", "professional", "warm_tone", "clear_speech"]
        )
        
        profiles["af_nicole"] = VoiceProfile(
            voice_id="af_nicole",
            name="Nicole (American Female)",
            gender="female",
            language="english",
            accent="american",
            quality_grade="B-",
            description="Professional and articulate American female voice. Great for business applications.",
            recommended_for=["professional", "business", "articulate", "formal"]
        )
        
        profiles["af_heart"] = VoiceProfile(
            voice_id="af_heart",
            name="Heart (American Female)",
            gender="female",
            language="english",
            accent="american",
            quality_grade="A",
            description="Premium quality American female voice with excellent naturalness.",
            recommended_for=["premium_quality", "natural", "conversational", "high_clarity"]
        )
        
        profiles["af_sarah"] = VoiceProfile(
            voice_id="af_sarah",
            name="Sarah (American Female)",
            gender="female",
            language="english",
            accent="american",
            quality_grade="C+",
            description="Casual and approachable American female voice.",
            recommended_for=["casual", "friendly", "approachable", "everyday"]
        )
        
        # Hindi Male Voices (for reference)
        profiles["hm_omega"] = VoiceProfile(
            voice_id="hm_omega",
            name="Arjun (Hindi Male)",
            gender="male",
            language="hindi",
            accent="indian",
            quality_grade="C",
            description="Hindi male voice with Indian accent.",
            recommended_for=["male_voice", "indian_accent", "hindi"]
        )
        
        return profiles
    
    def get_indian_female_voices(self) -> List[VoiceProfile]:
        """Get all Indian female voices"""
        return [
            profile for profile in self.profiles.values()
            if profile.gender == "female" and profile.accent == "indian"
        ]
    
    def get_high_quality_female_voices(self) -> List[VoiceProfile]:
        """Get high-quality female voices"""
        return [
            profile for profile in self.profiles.values()
            if profile.gender == "female" and profile.quality_grade in ["A", "A-", "B+", "B"]
        ]
    
    def get_recommended_voice_for_indian_accent(self) -> str:
        """Get the best voice for Indian accent"""
        indian_voices = self.get_indian_female_voices()
        if indian_voices:
            # Return the first Hindi female voice (hf_alpha)
            return indian_voices[0].voice_id
        else:
            # Fallback to high-quality English female
            return "af_bella"
    
    def get_voice_profile(self, voice_id: str) -> VoiceProfile:
        """Get profile for a specific voice"""
        return self.profiles.get(voice_id)
    
    def get_voice_recommendations(self, preference: str) -> List[str]:
        """Get voice recommendations based on preference"""
        recommendations = []
        
        if preference == "indian_female":
            recommendations = [v.voice_id for v in self.get_indian_female_voices()]
            recommendations.extend(["af_bella", "af_nicole"])  # High-quality fallbacks
        elif preference == "high_quality":
            recommendations = [v.voice_id for v in self.get_high_quality_female_voices()]
        elif preference == "professional":
            recommendations = ["af_nicole", "af_bella", "hf_alpha"]
        elif preference == "warm":
            recommendations = ["af_bella", "hf_beta", "af_sarah"]
        elif preference == "clear":
            recommendations = ["af_heart", "af_nicole", "hf_alpha"]
        else:
            # Default: Indian accent preference
            recommendations = ["hf_alpha", "hf_beta", "af_bella", "af_nicole"]
        
        return recommendations
    
    def get_voice_info_for_ui(self) -> Dict[str, Any]:
        """Get voice information formatted for UI display"""
        ui_info = {
            "indian_female": [],
            "english_female": [],
            "all_voices": []
        }
        
        for voice_id, profile in self.profiles.items():
            voice_info = {
                "id": voice_id,
                "name": profile.name,
                "description": profile.description,
                "quality": profile.quality_grade,
                "gender": profile.gender,
                "accent": profile.accent,
                "recommended": "indian_accent" in profile.recommended_for
            }
            
            ui_info["all_voices"].append(voice_info)
            
            if profile.gender == "female" and profile.accent == "indian":
                ui_info["indian_female"].append(voice_info)
            elif profile.gender == "female" and profile.language == "english":
                ui_info["english_female"].append(voice_info)
        
        return ui_info

# Global instance
voice_profile_manager = VoiceProfileManager()
