import logging
from typing import List

from vllm.entrypoints.openai.hb_serve import const

import prometheus_client
# from prometheus_client import Counter, CollectorRegistry

prometheus_client.disable_created_metrics()


class Metrics:
    labelname_fail_reason = "fail_reason"
    _base_library = prometheus_client

    def __init__(self, labelnames: List[str]):

        self._unregister_vllm_metrics()
        # self.registry = CollectorRegistry()

        # Config Information
        self.info_hb_serve_version = prometheus_client.Info(
            name='hb_serve:version_info',
            documentation='information of version')

        # System stats
        #   Requests State
        self.request_metrics_dict = {}

        self.counter_request_total = self._base_library.Counter(
            name="hb_serve:request_total",
            documentation="Count of processed requests.",
            labelnames=labelnames)
        self.request_metrics_dict[const.HBServeStatus.BAD_TOKEN_LENGTH.value] = self._base_library.Counter(
            name="hb_serve:request_fail_length",
            documentation="Count of processed fail requests with length.",
            labelnames=labelnames)
        self.request_metrics_dict[const.HBServeStatus.BAD_MODEL.value] = self._base_library.Counter(
            name="hb_serve:request_fail_model",
            documentation="Count of processed fail requests with model.",
            labelnames=labelnames)
        self.request_metrics_dict[const.HBServeStatus.BAD_GUIDED.value] = self._base_library.Counter(
            name="hb_serve:request_fail_guided",
            documentation="Count of processed fail requests with guided.",
            labelnames=labelnames)
        self.request_metrics_dict[const.HBServeStatus.BAD_REQUEST.value] = self._base_library.Counter(
            name="hb_serve:request_fail_other",
            documentation="Count of processed fail requests with other.",
            labelnames=labelnames)
        self.request_metrics_dict[const.HBServeStatus.BAD_HEADER.value] = self._base_library.Counter(
            name="hb_serve:request_fail_header",
            documentation="Count of processed fail requests with header.",
            labelnames=labelnames)
        self.request_metrics_dict[const.HBServeStatus.BAD_INPUT_PARAMS.value] = self._base_library.Counter(
            name="hb_serve:request_fail_input_params",
            documentation="Count of processed fail requests with input params.",
            labelnames=labelnames)
        self.request_metrics_dict[const.HBServeStatus.BAD_CONNECTION.value] = self._base_library.Counter(
            name="hb_serve:request_fail_connection",
            documentation="Count of processed fail requests with connection.",
            labelnames=labelnames)

    def _unregister_vllm_metrics(self) -> None:
        for collector in list(self._base_library.REGISTRY._collector_to_names):
            if hasattr(collector, "_name") and "hb_server" in collector._name:
                self._base_library.REGISTRY.unregister(collector)

    def init_counter(self, labels):
        self.counter_request_total.labels(**labels).inc(0)

        for status in const.HBServeStatus:
            if status in self.request_metrics_dict.keys():
                self.request_metrics_dict[status.value].labels(**labels).inc(0)

    def log(self, code=None, data=1, labels=None):
        try:
            if code is None:
                self.counter_request_total.labels(**labels).inc(max(0, data))
            else:
                self.request_metrics_dict[code].labels(**labels).inc(max(0, data))
        except:
            logging.error("log error")
