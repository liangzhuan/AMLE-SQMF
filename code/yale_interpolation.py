from yale_common import argument_parser, run_experiment


def main():
    args = argument_parser().parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
