from frey_common import (
    P_VALUES, argument_parser, prepare, query_centers, run_jobs, write_csv,
)


SAMPLE_SIZES = (30, 60, 90)


def main():
    parser = argument_parser("results_frey_sample_size")
    parser.add_argument("--queries", type=int, default=20)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--sizes", type=int, nargs="+", default=SAMPLE_SIZES)
    args = parser.parse_args()
    clean = prepare(args)
    centers = query_centers(args.queries, clean.shape[1])
    write_csv(args.output / "queries.csv", [
        dict(query_order=i, face_id=int(query) + 1)
        for i, query in enumerate(centers)
    ])
    jobs = [(index, args.alpha, order, int(query), n, args)
            for index in range(len(P_VALUES))
            for order, query in enumerate(centers)
            for n in args.sizes]
    run_jobs(jobs, args)


if __name__ == "__main__":
    main()
