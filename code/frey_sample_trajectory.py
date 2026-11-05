from frey_common import P_VALUES, argument_parser, prepare, run_jobs


def main():
    parser = argument_parser("results_frey_sample_trajectory")
    parser.add_argument("--query", type=int, default=1072)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--n-min", type=int, default=10)
    parser.add_argument("--n-max", type=int, default=100)
    parser.add_argument("--n-step", type=int, default=1)
    args = parser.parse_args()
    prepare(args)
    print(f"Trajectory query: {args.query}; alpha: {args.alpha:g}")
    jobs = [(index, args.alpha, 0, args.query - 1, n, args)
            for index in range(len(P_VALUES))
            for n in range(args.n_min, args.n_max + 1, args.n_step)]
    run_jobs(jobs, args, summary=False)


if __name__ == "__main__":
    main()
