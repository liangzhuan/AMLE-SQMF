from isomap_common import argument_parser, prepare, run_jobs


def main():
    parser = argument_parser()
    args = parser.parse_args()
    centers = prepare(args)
    run_jobs(centers, args)


if __name__ == "__main__":
    main()
