import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'src'))

from bidflow.extraction.chains import G3Chain
from bidflow.domain.models import CompanyProfile, ComplianceMatrix, ExtractionSlot
from bidflow.validation.validator import RuleBasedValidator

def test_extraction_and_validation():
    print("1. Mocking Extraction Context...")
    # 가상의 RFP 텍스트 (지역제한: 서울, 신용등급: B- 이상)
    context_text = """
    [입찰 참가 자격]
    1. 소프트웨어사업자(컴퓨터관련서비스사업) 신고를 필한 업체
    2. 주된 영업소의 소재지가 서울특별시인 업체
    3. 기업신용평가등급 B- 이상인 업체
    4. 공동수급 불허
    """
    
    print("2. Running G3Chain (Extraction)...")
    chain = G3Chain(model_name="gpt-5-mini")
    try:
        # G3Chain run arguments: context_text, project_name, issuer
        result = chain.run(context_text, "테스트 사업", "테스트 발주처")
        print(">> Extraction Result:")
        print(result.model_dump_json(indent=2))
        
        # Check Fields
        if not result.region_restriction.value:
            print("❌ Region Restriction Extracted Failed")
        else:
            print(f"✅ Region Extracted: {result.region_restriction.value}")
            
        if not result.financial_credit.value:
            print("❌ Credit Rating Extracted Failed")
        else:
            print(f"✅ Credit Rating Extracted: {result.financial_credit.value}")

    except Exception as e:
        print(f"❌ Chain Execution Failed: {e}")
        return

    print("\n3. Testing Validation Logic...")
    validator = RuleBasedValidator()
    
    # Case A: Pass Profile
    profile_pass = CompanyProfile(
        id="p1", name="Pass Corp", 
        data={"region": "서울", "credit_rating": "A", "licenses": ["소프트웨어사업자"]}
    )
    
    # Construct Matrix
    matrix = ComplianceMatrix(
        doc_hash="test",
        slots={
            "required_licenses": result.required_licenses,
            "region_restriction": result.region_restriction,
            "financial_credit": result.financial_credit,
            "restrictions": result.restrictions
        }
    )
    
    res_pass = validator.validate(matrix, profile_pass)
    print(f">> Profile (Region=서울, Credit=A): {[r.decision for r in res_pass]}")
    
    # Verify Decisions
    # We expect GREEN for Region, GREEN for Credit
    for r in res_pass:
        if r.slot_key == "region_restriction":
            assert r.decision == "GREEN", f"Region check failed for Pass Profile: {r.reasons}"
        if r.slot_key == "financial_credit":
            assert r.decision == "GREEN", f"Credit check failed for Pass Profile: {r.reasons}"

    # Case B: Fail Profile
    profile_fail = CompanyProfile(
        id="p2", name="Fail Corp", 
        data={"region": "부산", "credit_rating": "CCC", "licenses": []}
    )
    res_fail = validator.validate(matrix, profile_fail)
    print(f">> Profile (Region=부산, Credit=CCC): {[r.decision for r in res_fail]}")

    # Verify Decisions
    # We expect RED for Region, RED for Credit
    for r in res_fail:
        if r.slot_key == "region_restriction":
            assert r.decision == "RED", f"Region check failed for Fail Profile (Expected RED): {r.reasons}"
        if r.slot_key == "financial_credit":
            assert r.decision == "RED", f"Credit check failed for Fail Profile (Expected RED): {r.reasons}"
            
    print("\n✅ All Tests Passed!")

if __name__ == "__main__":
    test_extraction_and_validation()
