# gradio_app.py

import gradio as gr
import requests

API_URL = "https://10.20.152.74:8200"


def fetch_logs(date, prefix):
    res = requests.get(f"{API_URL}/logs", params={"data": date, "prefix": prefix},verify=False)
    if res.status_code != 200:
        return [], [], "错误：无法获取日志"
    data = res.json()
    logs = data["curl"]
    responses = data["response"]
    return logs, responses, "已加载日志"


def load_contents(curl_name = None, response_name=None):
    if curl_name:
        curl = requests.get(f"{API_URL}/log_content", params={"object_name": curl_name},verify=False).json()["content"]
    else:
        curl = ""
    if response_name:
        resp = requests.get(f"{API_URL}/log_content", params={"object_name": response_name},verify=False).json()["content"]
    else:
        resp = ""
    return curl, resp




def run_replay(curl_name, host, port):
    res = requests.post(f"{API_URL}/replay", params={"object_name": curl_name, "host": host, "port": port},verify=False)
    return res.json().get("message", "执行失败")

def curl_and_response_process(date, prefix, selected_index):
    logs, responses, _ = fetch_logs(date, prefix)
    selected_index = 0
    if not logs and not responses:
        return "", "", f"未找到 Request ID {prefix} 的日志"

    if not logs or selected_index >= len(logs):
        curl, response = load_contents(response_name=responses[selected_index])
        return curl, response, "未找到curl日志"

    if not responses or selected_index >= len(responses):
        curl, response = load_contents(curl_name=logs[selected_index])
        return curl, response, "未找到respone日志"

    curl, response = load_contents(logs[selected_index], responses[selected_index])
    return curl, response, "获取curl和response完成"

def replay_process(date, prefix, selected_index, host, port):
    logs, responses, _ = fetch_logs(date, prefix)
    selected_index = 0
    # replay_result = run_replay(logs[selected_index], host, port)
    # return replay_result, "回放完成"
    for res in stream_replay_output(logs[selected_index], host, port):
        yield res, "回放进行中"
    return "", "回放完成"


def full_process(date, prefix, selected_index, host, port):
    logs, responses, _ = fetch_logs(date, prefix)
    selected_index = 0
    if not logs or selected_index >= len(logs):
        return "", "", "", "未找到日志"
    curl, response = load_contents(logs[selected_index], responses[selected_index])
    replay_result = run_replay(logs[selected_index], host, port)
    return curl, response, replay_result, "回放完成"


def stream_replay_output(curl_name, host, port):
    params = {"object_name": curl_name}
    if host:
        params["host"] = host
    if port:
        params["port"] = port
    accumulated = ""
    with requests.post(f"{API_URL}/replay", params=params, stream=True, verify=False) as r:
        try:
            for line in r.iter_lines(decode_unicode=True):
                if line:
                    accumulated += line + "\n"
                    yield accumulated
        except Exception as e:
            accumulated += "data: [DONE]"
            yield accumulated


with gr.Blocks() as demo:
    with gr.Row():
        date = gr.Textbox(label="Date (YYYY-MM-DD)")
        prefix = gr.Textbox(label="Request_ID")
        host = gr.Textbox(label="Host")
        port = gr.Textbox(label="Port")
        refresh_btn = gr.Button("加载日志")

    logs_list = gr.Dropdown(label="选择日志项", choices=[], interactive=True)
    status = gr.Textbox(label="状态", interactive=False)

    with gr.Row():
        curl_text = gr.Textbox(label="CURL 内容", lines=15)
        orig_resp = gr.Textbox(label="原始响应", lines=15)
        replay_resp = gr.Textbox(label="回放结果", lines=15, interactive=False)
    with gr.Row():
        curl_text_btn = gr.Button("获取 CURL 内容及原始响应")
        replay_resp_btn = gr.Button("获取回放")

    run_btn = gr.Button("一键回放")


    # 交互绑定
    def update_log_choices(date, prefix):
        logs, _, stat = fetch_logs(date, prefix)
        return gr.update(choices=logs), stat


    refresh_btn.click(fn=update_log_choices, inputs=[date, prefix], outputs=[logs_list, status])

    curl_text_btn.click(
        fn=curl_and_response_process,
        inputs=[date, prefix, logs_list],
        outputs=[curl_text, orig_resp,status]
    )

    replay_resp_btn.click(
        fn=replay_process,
        inputs=[date, prefix, logs_list, host, port],
        outputs=[replay_resp, status]
    )


    run_btn.click(
        fn=full_process,
        inputs=[date, prefix, logs_list, host, port],
        outputs=[curl_text, orig_resp, replay_resp, status]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=6001)
