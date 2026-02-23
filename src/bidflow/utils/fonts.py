import platform
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os

def set_korean_font():
    """
    OS에 따른 matplotlib 한글 폰트 설정
    """
    system_name = platform.system()
    
    if system_name == "Windows":
        # Windows: 맑은 고딕
        font_path = "C:/Windows/Fonts/malgun.ttf"
        font_name = "Malgun Gothic"
    elif system_name == "Darwin":
        # Mac: 애플고딕
        font_path = "/Library/Fonts/AppleGothic.ttf"
        font_name = "AppleGothic"
    else:
        # Linux: 나눔고딕 (설치 필요)
        # 우선순위: NanumGothic -> Malgun Gothic -> AppleGothic
        font_list = [f.name for f in fm.fontManager.ttflist]
        if "NanumGothic" in font_list:
            font_name = "NanumGothic"
        else:
            font_name = "DejaVu Sans" # Fallback

    try:
        if system_name == "Windows" and os.path.exists(font_path):
            font_name = fm.FontProperties(fname=font_path).get_name()
            plt.rc("font", family=font_name)
        else:
            plt.rc("font", family=font_name)
            
        plt.rc("axes", unicode_minus=False) # 마이너스 기호 깨짐 방지
        print(f"[Font] Matplotlib font set to: {font_name}")
        return True
    except Exception as e:
        print(f"[Font] Failed to set font: {e}")
        return False
