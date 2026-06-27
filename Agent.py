from model import txt_sending, save_to_session

def talk_to_model(prompt: str, override: str | None = None):
    result = txt_sending(prompt=prompt, override=override)
    save_to_session(prompt, result)                         # ← saves every response automatically
    return result