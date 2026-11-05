from frey_common import (
    P_VALUES, argument_parser, prepare, query_centers, run_jobs, write_csv,
)


ALPHA_VALUES = (0.05, 0.10, 0.15, 0.20)


def main():
    parser = argument_parser("results_frey_noise_scale")
    parser.add_argument("--queries", type=int, default=30)
    parser.add_argument("--n", type=int, default=100)
    args = parser.parse_args()
    clean = prepare(args)
    centers = query_centers(args.queries, clean.shape[1])
    write_csv(args.output / "queries.csv", [
        dict(query_order=i, face_id=int(query) + 1)
        for i, query in enumerate(centers)
    ])
    jobs = [(index, alpha, order, int(query), args.n, args)
            for alpha in ALPHA_VALUES
            for index in range(len(P_VALUES))
            for order, query in enumerate(centers)]
    run_jobs(jobs, args)


if __name__ == "__main__":
    main()
