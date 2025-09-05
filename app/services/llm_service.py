"""
LLM Service for Advanced Policy Analysis and Rule Extraction
Supports multiple LLM providers with fallback and retry logic
"""
import os
import json
import asyncio
import time
from typing import List, Dict, Any, Optional, Union, AsyncGenerator
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import openai
import anthropic
from dataclasses import dataclass

from app.config.settings import get_settings
from app.utils.logger import get_logger, log_execution_time
from app.utils.exceptions import LLMServiceException, ValidationException

settings = get_settings()
logger = get_logger(__name__)


@dataclass
class LLMResponse:
    """Standardized LLM response structure"""
    content: str
    model: str
    provider: str
    tokens_used: Optional[int] = None
    processing_time: Optional[float] = None
    confidence: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


class LLMProvider:
    """Base class for LLM providers"""
    
    def __init__(self, name: str, api_key: str, model: str):
        self.name = name
        self.api_key = api_key
        self.model = model
        self.available = bool(api_key)
    
    async def generate(self, prompt: str, **kwargs) -> LLMResponse:
        raise NotImplementedError


class OpenAIProvider(LLMProvider):
    """OpenAI GPT provider"""
    
    def __init__(self, api_key: str, model: str = "gpt-4"):
        super().__init__("openai", api_key, model)
        if self.available:
            self.client = openai.OpenAI(api_key=api_key)
    
    async def generate(self, prompt: str, 
                      temperature: float = 0.1,
                      max_tokens: int = 2000,
                      **kwargs) -> LLMResponse:
        """Generate response using OpenAI API"""
        try:
            start_time = time.time()
            
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": kwargs.get("system_message", 
                                "You are an expert compliance analyst.")
                        },
                        {"role": "user", "content": prompt}
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens
                )
            )
            
            processing_time = time.time() - start_time
            content = response.choices[0].message.content
            tokens_used = response.usage.total_tokens
            
            return LLMResponse(
                content=content,
                model=self.model,
                provider=self.name,
                tokens_used=tokens_used,
                processing_time=processing_time,
                metadata={
                    "finish_reason": response.choices[0].finish_reason,
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens
                }
            )
            
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise LLMServiceException(f"OpenAI generation failed: {str(e)}")


class AnthropicProvider(LLMProvider):
    """Anthropic Claude provider"""
    
    def __init__(self, api_key: str, model: str = "claude-3-haiku-20240307"):
        super().__init__("anthropic", api_key, model)
        if self.available:
            self.client = anthropic.Anthropic(api_key=api_key)
    
    async def generate(self, prompt: str,
                      temperature: float = 0.1,
                      max_tokens: int = 2000,
                      **kwargs) -> LLMResponse:
        """Generate response using Anthropic API"""
        try:
            start_time = time.time()
            
            message = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    messages=[{
                        "role": "user", 
                        "content": prompt
                    }]
                )
            )
            
            processing_time = time.time() - start_time
            content = message.content[0].text
            
            return LLMResponse(
                content=content,
                model=self.model,
                provider=self.name,
                processing_time=processing_time,
                metadata={
                    "usage": message.usage.__dict__ if hasattr(message, 'usage') else None
                }
            )
            
        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            raise LLMServiceException(f"Anthropic generation failed: {str(e)}")


class MockLLMProvider(LLMProvider):
    """Mock LLM provider for testing and development"""
    
    def __init__(self):
        super().__init__("mock", "mock-key", "mock-model")
        self.available = True
    
    async def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate mock response"""
        await asyncio.sleep(0.1)  # Simulate processing time
        
        # Generate mock compliance rules based on prompt content
        if "extract compliance rules" in prompt.lower():
            mock_rules = [
                {
                    "rule_id": "mock-concentration-limit",
                    "description": "Mock issuer concentration limit rule",
                    "control_type": "quant_limit",
                    "severity": "high",
                    "expression": {
                        "metric": "issuer_weight",
                        "operator": "<=",
                        "threshold": 0.05,
                        "scope": "portfolio"
                    },
                    "confidence": 0.85,
                    "source_section": "Mock Section 1.1"
                }
            ]
            content = json.dumps(mock_rules, indent=2)
        else:
            content = "This is a mock LLM response for development and testing purposes."
        
        return LLMResponse(
            content=content,
            model="mock-model",
            provider="mock",
            processing_time=0.1,
            confidence=0.8,
            metadata={"is_mock": True}
        )


class LLMService:
    """Advanced LLM service with multiple providers and intelligent routing"""
    
    def __init__(self):
        self.providers = {}
        self.default_provider = None
        self.max_retries = 3
        self.retry_delay = 1.0
        self.logger = get_logger("llm_service")
        
        self._initialize_providers()
    
    def _initialize_providers(self):
        """Initialize available LLM providers"""
        # Initialize OpenAI provider
        if settings.openai_api_key:
            try:
                self.providers["openai"] = OpenAIProvider(
                    settings.openai_api_key, 
                    settings.llm_model
                )
                if not self.default_provider:
                    self.default_provider = "openai"
                self.logger.info("OpenAI provider initialized")
            except Exception as e:
                self.logger.warning(f"Failed to initialize OpenAI provider: {e}")
        
        # Initialize Anthropic provider
        if settings.anthropic_api_key:
            try:
                self.providers["anthropic"] = AnthropicProvider(
                    settings.anthropic_api_key
                )
                if not self.default_provider:
                    self.default_provider = "anthropic"
                self.logger.info("Anthropic provider initialized")
            except Exception as e:
                self.logger.warning(f"Failed to initialize Anthropic provider: {e}")
        
        # Initialize mock provider for testing
        if settings.mock_llm or not self.providers:
            self.providers["mock"] = MockLLMProvider()
            if not self.default_provider:
                self.default_provider = "mock"
            self.logger.info("Mock LLM provider initialized")
        
        if not self.providers:
            raise LLMServiceException("No LLM providers available")
    
    async def generate(self, prompt: str, provider: str = None, 
                      retry_on_failure: bool = True, **kwargs) -> LLMResponse:
        """Generate response with automatic retry and fallback"""
        target_provider = provider or self.default_provider
        
        if target_provider not in self.providers:
            raise LLMServiceException(f"Provider {target_provider} not available")
        
        last_exception = None
        
        # Try primary provider with retries
        for attempt in range(self.max_retries):
            try:
                return await self.providers[target_provider].generate(prompt, **kwargs)
            except Exception as e:
                last_exception = e
                self.logger.warning(
                    f"Attempt {attempt + 1} failed for provider {target_provider}: {e}"
                )
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay * (2 ** attempt))
        
        # Try fallback providers if retry is enabled
        if retry_on_failure and len(self.providers) > 1:
            for fallback_provider in self.providers:
                if fallback_provider != target_provider:
                    try:
                        self.logger.info(f"Trying fallback provider: {fallback_provider}")
                        return await self.providers[fallback_provider].generate(prompt, **kwargs)
                    except Exception as e:
                        self.logger.warning(f"Fallback provider {fallback_provider} failed: {e}")
                        continue
        
        raise LLMServiceException(f"All providers failed. Last error: {str(last_exception)}")
    
    @log_execution_time("extract_rules_from_chunks")
    async def extract_rules_from_chunks(self, chunks: List[Dict[str, Any]], 
                                      policy_metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract compliance rules from policy chunks using advanced prompting
        """
        try:
            if not chunks:
                return []
            
            # Process chunks in batches for efficiency
            batch_size = 3
            all_rules = []
            
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                batch_rules = await self._process_chunk_batch(batch, policy_metadata)
                all_rules.extend(batch_rules)
            
            # Post-process and deduplicate rules
            processed_rules = self._post_process_extracted_rules(all_rules, policy_metadata)
            
            self.logger.info(
                f"Extracted {len(processed_rules)} rules from {len(chunks)} chunks"
            )
            
            return processed_rules
            
        except Exception as e:
            self.logger.error(f"Error in rule extraction: {e}")
            return self._fallback_rule_extraction(chunks, policy_metadata)
    
    async def _process_chunk_batch(self, chunks: List[Dict[str, Any]], 
                                 policy_metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Process a batch of chunks for rule extraction"""
        try:
            # Combine chunks for context
            combined_content = "\n\n".join([
                f"[Section: {chunk.get('section_title', 'Unknown')}]\n{chunk['content']}"
                for chunk in chunks
            ])
            
            prompt = self._build_advanced_extraction_prompt(
                combined_content, 
                policy_metadata,
                chunks[0].get('section_title', 'Unknown')
            )
            
            response = await self.generate(
                prompt,
                temperature=0.05,  # Low temperature for consistent rule extraction
                max_tokens=3000,
                system_message="You are an expert compliance analyst specializing in extracting structured compliance rules from policy documents."
            )
            
            return self._parse_llm_response(response.content, chunks, policy_metadata)
            
        except Exception as e:
            self.logger.error(f"Error processing chunk batch: {e}")
            return []
    
    def _build_advanced_extraction_prompt(self, content: str, policy_metadata: Dict[str, Any], 
                                        section_title: str) -> str:
        """Build advanced prompt for rule extraction"""
        
        prompt = f"""
Extract concrete, measurable compliance rules from the following policy text. Focus on identifying rules that can be programmatically monitored and enforced.

POLICY CONTEXT:
- Document Type: {policy_metadata.get('document_type', 'Policy Document')}
- Jurisdiction: {policy_metadata.get('jurisdiction', 'Unknown')}
- Section: {section_title}

CONTENT TO ANALYZE:
{content}

EXTRACTION GUIDELINES:
1. Only extract rules that are specific and actionable
2. Focus on quantitative limits, prohibited/allowed lists, time constraints, process requirements, and reporting obligations
3. Ignore general principles or aspirational statements
4. Each rule must be independently enforceable

For each rule found, return a JSON object with this EXACT structure:

{{
    "rule_id": "descriptive-kebab-case-identifier",
    "description": "Clear, concise description of what the rule requires or prohibits",
    "control_type": "quant_limit|list_constraint|temporal_window|process_control|reporting_disclosure",
    "severity": "critical|high|medium|low",
    "expression": {{
        // Control-type specific parameters:
        
        // For quant_limit (quantitative thresholds):
        "metric": "issuer_weight|sector_weight|country_weight|total_exposure|leverage_ratio|rating_weight",
        "operator": "<=|>=|<|>|==|!=",
        "threshold": 0.05,  // numeric value
        "scope": "portfolio|position|trade",
        "group_by": "issuer|sector|country|rating",  // optional
        "filter": "specific_value_to_filter_by",  // optional
        
        // For list_constraint (allowed/prohibited values):
        "field": "rating|sector|country|issuer|instrument_type",
        "allowed_values": ["AAA", "AA+", "AA"],  // optional
        "denied_values": ["C", "D"],  // optional
        "scope": "portfolio|position",
        
        // For temporal_window (time-based constraints):
        "metric": "holding_period|lock_up_period|settlement_period",
        "minimum_days": 30,
        "maximum_days": 365,  // optional
        "scope": "position|trade",
        
        // For process_control (approval/workflow requirements):
        "approval_required": true,
        "approver_role": "risk_manager|compliance_officer|cio",
        "evidence_required": "Description of required documentation",
        "sla_days": 5,
        
        // For reporting_disclosure (reporting obligations):
        "report_type": "position_report|risk_report|breach_report|regulatory_filing",
        "frequency": "daily|weekly|monthly|quarterly|annually",
        "deadline_days": 30,
        "recipient": "regulator|board|management|public"
    }},
    "materiality_bps": 100,  // basis points threshold for materiality (optional, default 0)
    "source_section": "{section_title}",
    "confidence": 0.85,  // 0.0-1.0 confidence in rule extraction
    "rationale": "Brief explanation of why this was identified as a rule"
}}

SEVERITY GUIDELINES:
- critical: Regulatory violations, major risk limits
- high: Significant operational controls, important thresholds
- medium: Standard procedures, moderate risk controls
- low: Minor administrative requirements

CONFIDENCE GUIDELINES:
- 0.9-1.0: Explicit rule with clear parameters
- 0.7-0.9: Clear rule but some interpretation required
- 0.5-0.7: Inferred rule from guidance
- 0.0-0.5: Uncertain or vague rule

Return ONLY a JSON array of rule objects. If no extractable rules are found, return an empty array [].

Example response format:
[
    {{
        "rule_id": "single-issuer-limit",
        "description": "No single issuer shall exceed 5% of portfolio value",
        "control_type": "quant_limit",
        "severity": "critical",
        "expression": {{
            "metric": "issuer_weight",
            "operator": "<=",
            "threshold": 0.05,
            "scope": "portfolio"
        }},
        "materiality_bps": 50,
        "source_section": "{section_title}",
        "confidence": 0.95,
        "rationale": "Explicit concentration limit stated in policy"
    }}
]
"""
        
        return prompt
    
    def _parse_llm_response(self, response_text: str, chunks: List[Dict[str, Any]], 
                          policy_metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Parse and validate LLM response"""
        try:
            # Clean response text
            response_text = response_text.strip()
            
            # Extract JSON array
            if response_text.startswith('[') and response_text.endswith(']'):
                rules = json.loads(response_text)
            else:
                # Try to find JSON array in text
                import re
                json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
                if json_match:
                    rules = json.loads(json_match.group())
                else:
                    self.logger.warning("No valid JSON array found in LLM response")
                    return []
            
            if not isinstance(rules, list):
                return []
            
            # Validate and enhance each rule
            validated_rules = []
            for rule in rules:
                try:
                    if self._validate_extracted_rule(rule):
                        # Add extraction metadata
                        rule["extraction_metadata"] = {
                            "extracted_at": datetime.now().isoformat(),
                            "extraction_method": "llm_advanced",
                            "chunk_sources": [chunk.get("chunk_id") for chunk in chunks],
                            "policy_context": policy_metadata
                        }
                        validated_rules.append(rule)
                    else:
                        self.logger.warning(f"Invalid rule structure: {rule.get('rule_id', 'unknown')}")
                except Exception as e:
                    self.logger.warning(f"Error validating rule: {e}")
                    continue
            
            return validated_rules
            
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON parsing error in LLM response: {e}")
            self.logger.debug(f"Response text: {response_text[:500]}...")
            return []
        except Exception as e:
            self.logger.error(f"Error parsing LLM response: {e}")
            return []
    
    def _validate_extracted_rule(self, rule: Dict[str, Any]) -> bool:
        """Validate extracted rule structure and content"""
        try:
            # Check required fields
            required_fields = ["rule_id", "description", "control_type", "severity", "expression"]
            if not all(field in rule for field in required_fields):
                return False
            
            # Validate control_type
            valid_control_types = [
                "quant_limit", "list_constraint", "temporal_window", 
                "process_control", "reporting_disclosure"
            ]
            if rule["control_type"] not in valid_control_types:
                return False
            
            # Validate severity
            valid_severities = ["critical", "high", "medium", "low"]
            if rule["severity"] not in valid_severities:
                return False
            
            # Validate expression based on control type
            expression = rule["expression"]
            if not isinstance(expression, dict):
                return False
            
            control_type = rule["control_type"]
            
            # Control-type specific validation
            if control_type == "quant_limit":
                required_expr_fields = ["metric", "operator", "threshold", "scope"]
                if not all(field in expression for field in required_expr_fields):
                    return False
                
                valid_operators = ["<=", ">=", "<", ">", "==", "!="]
                if expression["operator"] not in valid_operators:
                    return False
                
                if not isinstance(expression["threshold"], (int, float)):
                    return False
            
            elif control_type == "list_constraint":
                required_expr_fields = ["field", "scope"]
                if not all(field in expression for field in required_expr_fields):
                    return False
                
                if not ("allowed_values" in expression or "denied_values" in expression):
                    return False
            
            elif control_type == "temporal_window":
                required_expr_fields = ["metric", "minimum_days", "scope"]
                if not all(field in expression for field in required_expr_fields):
                    return False
                
                if not isinstance(expression["minimum_days"], int) or expression["minimum_days"] < 0:
                    return False
            
            elif control_type == "process_control":
                required_expr_fields = ["approval_required", "evidence_required"]
                if not all(field in expression for field in required_expr_fields):
                    return False
            
            elif control_type == "reporting_disclosure":
                required_expr_fields = ["report_type", "frequency"]
                if not all(field in expression for field in required_expr_fields):
                    return False
            
            # Validate confidence if present
            if "confidence" in rule:
                confidence = rule["confidence"]
                if not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Rule validation error: {e}")
            return False
    
    def _post_process_extracted_rules(self, rules: List[Dict[str, Any]], 
                                    policy_metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Post-process extracted rules to deduplicate and enhance"""
        try:
            if not rules:
                return []
            
            # Deduplicate based on rule_id and similarity
            unique_rules = {}
            
            for rule in rules:
                rule_id = rule.get("rule_id", f"rule_{len(unique_rules)}")
                
                if rule_id in unique_rules:
                    # Keep rule with higher confidence
                    existing_confidence = unique_rules[rule_id].get("confidence", 0)
                    new_confidence = rule.get("confidence", 0)
                    
                    if new_confidence > existing_confidence:
                        unique_rules[rule_id] = rule
                else:
                    unique_rules[rule_id] = rule
            
            # Sort by confidence and severity
            severity_order = {"critical": 4, "high": 3, "medium": 2, "low": 1}
            
            processed_rules = list(unique_rules.values())
            processed_rules.sort(
                key=lambda r: (
                    severity_order.get(r.get("severity", "low"), 1),
                    r.get("confidence", 0)
                ), 
                reverse=True
            )
            
            # Add sequential IDs if missing
            for i, rule in enumerate(processed_rules):
                if not rule.get("rule_id") or rule["rule_id"].startswith("rule_"):
                    rule["rule_id"] = f"extracted-rule-{i+1}"
            
            return processed_rules
            
        except Exception as e:
            self.logger.error(f"Error post-processing rules: {e}")
            return rules
    
    def _fallback_rule_extraction(self, chunks: List[Dict[str, Any]], 
                                 policy_metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Fallback rule extraction using pattern matching"""
        self.logger.info("Using fallback pattern-based rule extraction")
        
        rules = []
        
        for chunk in chunks:
            content = chunk["content"].lower()
            
            # Pattern for percentage limits
            import re
            percentage_patterns = [
                r"(?:shall not exceed|cannot exceed|limited to|maximum of)\s+(\d+(?:\.\d+)?)\s*%",
                r"(\d+(?:\.\d+)?)\s*%\s+(?:limit|maximum|cap|threshold)"
            ]
            
            for pattern in percentage_patterns:
                matches = re.finditer(pattern, content)
                for match in matches:
                    try:
                        threshold = float(match.group(1)) / 100.0
                        
                        rules.append({
                            "rule_id": f"pattern-extracted-{len(rules)+1}",
                            "description": f"Portfolio exposure limit of {threshold*100}%",
                            "control_type": "quant_limit",
                            "severity": "medium",
                            "expression": {
                                "metric": "issuer_weight",
                                "operator": "<=",
                                "threshold": threshold,
                                "scope": "portfolio"
                            },
                            "confidence": 0.6,
                            "source_section": chunk.get("section_title", "Unknown"),
                            "extraction_metadata": {
                                "extraction_method": "pattern_based_fallback",
                                "pattern_matched": pattern,
                                "extracted_at": datetime.now().isoformat()
                            }
                        })
                    except ValueError:
                        continue
            
            # Pattern for rating requirements
            rating_patterns = [
                r"(?:minimum|at least)\s+([A-Z]{1,4}[+-]?)\s+(?:rating|grade)",
                r"rated\s+([A-Z]{1,4}[+-]?)\s+or\s+(?:higher|above)"
            ]
            
            for pattern in rating_patterns:
                matches = re.finditer(pattern, content)
                for match in matches:
                    rating = match.group(1).upper()
                    
                    rules.append({
                        "rule_id": f"rating-requirement-{len(rules)+1}",
                        "description": f"Minimum credit rating requirement: {rating}",
                        "control_type": "list_constraint",
                        "severity": "medium",
                        "expression": {
                            "field": "rating",
                            "allowed_values": self._get_allowed_ratings(rating),
                            "scope": "position"
                        },
                        "confidence": 0.7,
                        "source_section": chunk.get("section_title", "Unknown"),
                        "extraction_metadata": {
                            "extraction_method": "pattern_based_fallback",
                            "pattern_matched": pattern,
                            "extracted_at": datetime.now().isoformat()
                        }
                    })
        
        return rules
    
    def _get_allowed_ratings(self, minimum_rating: str) -> List[str]:
        """Get list of allowed credit ratings based on minimum requirement"""
        rating_scale = [
            "AAA", "AA+", "AA", "AA-", "A+", "A", "A-", 
            "BBB+", "BBB", "BBB-", "BB+", "BB", "BB-",
            "B+", "B", "B-", "CCC+", "CCC", "CCC-", "D"
        ]
        
        try:
            min_index = rating_scale.index(minimum_rating)
            return rating_scale[:min_index + 1]
        except ValueError:
            # Default to investment grade if rating not found
            return rating_scale[:10]
    
    async def generate_breach_explanation(self, breach_data: Dict[str, Any], 
                                        rule_data: Dict[str, Any],
                                        policy_context: List[Dict[str, Any]] = None) -> str:
        """Generate detailed breach explanation"""
        try:
            context_text = ""
            if policy_context:
                context_text = "\n".join([
                    f"Policy Context: {ctx['content'][:200]}..." 
                    for ctx in policy_context[:2]
                ])
            
            prompt = f"""
Generate a professional compliance breach explanation for regulatory reporting and management review.

BREACH DETAILS:
- Rule: {rule_data.get('description', 'Unknown rule')}
- Control Type: {rule_data.get('control_type', 'unknown')}
- Severity: {rule_data.get('severity', 'unknown')}
- Observed Value: {breach_data.get('observed_value', 'N/A')}
- Threshold: {breach_data.get('threshold_value', 'N/A')}
- Detection Time: {breach_data.get('breach_timestamp', 'N/A')}
- Breach Magnitude: {breach_data.get('breach_magnitude', 'N/A')}%

POLICY CONTEXT:
{context_text}

Generate a comprehensive explanation that includes:
1. Executive Summary (what happened)
2. Technical Details (specific values and calculations)
3. Root Cause Analysis (why it happened)
4. Impact Assessment (potential consequences)
5. Immediate Actions Required
6. Regulatory Implications

Use professional, precise language appropriate for compliance reporting.
Keep the explanation concise but thorough (400-600 words).
"""

            response = await self.generate(
                prompt,
                temperature=0.1,
                max_tokens=1000,
                system_message="You are a compliance officer generating breach explanations for regulatory reporting."
            )
            
            return response.content.strip()
            
        except Exception as e:
            self.logger.error(f"Error generating breach explanation: {e}")
            return self._fallback_breach_explanation(breach_data, rule_data)
    
    def _fallback_breach_explanation(self, breach_data: Dict[str, Any], 
                                   rule_data: Dict[str, Any]) -> str:
        """Fallback breach explanation without LLM"""
        
        return f"""
COMPLIANCE BREACH REPORT

Executive Summary:
A compliance breach has been detected for rule: {rule_data.get('description', 'Unknown rule')}

Technical Details:
- Control Type: {rule_data.get('control_type', 'unknown')}
- Severity Level: {rule_data.get('severity', 'unknown').upper()}
- Observed Value: {breach_data.get('observed_value', 'N/A')}
- Regulatory Threshold: {breach_data.get('threshold_value', 'N/A')}
- Breach Magnitude: {breach_data.get('breach_magnitude', 'N/A')}%
- Detection Timestamp: {breach_data.get('breach_timestamp', 'N/A')}

Impact Assessment:
The breach indicates portfolio positions currently exceed permitted limits as defined 
in the compliance framework. This may result in regulatory scrutiny and require 
immediate corrective action to restore compliance.

Immediate Actions Required:
1. Review current portfolio positions
2. Assess need for position adjustments
3. Document circumstances of breach
4. Notify relevant stakeholders
5. Implement corrective measures

This breach explanation was generated using fallback processing. 
Please refer to complete policy documentation and consult with the compliance team 
for detailed remediation procedures.

Generated: {datetime.now().isoformat()}
""".strip()
    
    async def analyze_policy_sentiment(self, policy_text: str) -> Dict[str, Any]:
        """Analyze policy text for compliance sentiment and complexity"""
        try:
            prompt = f"""
Analyze the following policy text for compliance characteristics:

{policy_text[:2000]}

Provide analysis in JSON format:
{{
    "complexity_score": 0.0-1.0,
    "enforceability_score": 0.0-1.0,
    "clarity_score": 0.0-1.0,
    "regulatory_intensity": "low|medium|high|critical",
    "key_themes": ["theme1", "theme2"],
    "potential_ambiguities": ["ambiguity1", "ambiguity2"],
    "recommended_actions": ["action1", "action2"]
}}
"""
            
            response = await self.generate(
                prompt,
                temperature=0.2,
                max_tokens=800
            )
            
            try:
                return json.loads(response.content)
            except json.JSONDecodeError:
                return {"error": "Failed to parse analysis response"}
                
        except Exception as e:
            self.logger.error(f"Error in policy sentiment analysis: {e}")
            return {"error": str(e)}
    
    def get_service_status(self) -> Dict[str, Any]:
        """Get comprehensive LLM service status"""
        provider_status = {}
        for name, provider in self.providers.items():
            provider_status[name] = {
                "available": provider.available,
                "model": provider.model,
                "is_default": name == self.default_provider
            }
        
        return {
            "service": "llm_service",
            "status": "ready" if self.providers else "unavailable",
            "default_provider": self.default_provider,
            "providers": provider_status,
            "max_retries": self.max_retries,
            "retry_delay": self.retry_delay,
            "features": {
                "rule_extraction": True,
                "breach_explanation": True,
                "policy_analysis": True,
                "multi_provider_fallback": len(self.providers) > 1
            }
        }