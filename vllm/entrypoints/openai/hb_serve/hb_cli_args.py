from vllm.utils.argparse_utils import FlexibleArgumentParser


def make_hb_arg_parser(parser: FlexibleArgumentParser) -> FlexibleArgumentParser:
    parser.add_argument("--allow-consul",
                        type=str,
                        default='false',
                        help="true or false.")
    parser.add_argument("--consul-host",
                        type=str,
                        default=None,
                        help="Consul host name.")
    parser.add_argument("--consul-port",
                        type=str,
                        default=None,
                        help="Consul port number.")
    parser.add_argument("--consul-token",
                        type=str,
                        default=None,
                        help="Consul token.")
    parser.add_argument("--token-pool-size",
                        type=int,
                        default=None,
                        help="Max acceptable token.")
    # todo remove
    parser.add_argument("--use-priority",
                        type=str,
                        default=None,
                        help="whether to use priority for schuduler")
    parser.add_argument("--version",
                        type=str,
                        default=None,
                        help="Serve version")
    parser.add_argument("--lora-models",
                        type=str,
                        default=None,
                        help="lora models all one")
    parser.add_argument("--is-reasoning-model",
                        type=str,
                        default="false",
                        help="Whether it is a model that supports reasoning")
    return parser

