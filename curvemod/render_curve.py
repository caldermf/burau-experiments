from __future__ import annotations

import argparse

from curve_renderer import RenderStyle, curve_pair_from_word, parse_artin_word, render_curve_pair_with_style


def main() -> None:
    parser = argparse.ArgumentParser(description="Render the curve pair (P_i, w(P_i)) for a word in A3 Artin generators.")
    parser.add_argument("--word", default="", help='Space- or comma-separated generators, e.g. "2 -1 3".')
    parser.add_argument("--base-arc", type=int, default=1, help="Base arc index. v1 supports 1, 2, or 3.")
    parser.add_argument("--out", default=None, help="Optional output image path.")
    parser.add_argument("--hide-base", action="store_true", help="Render only the moved curve w(P_i).")
    parser.add_argument("--moved-opacity", type=float, default=0.18, help="Per-segment opacity for the moved curve.")
    parser.add_argument("--moved-linewidth", type=float, default=0.8, help="Line width for the moved curve.")
    parser.add_argument("--no-title", action="store_true", help="Omit the title to maximize visible area.")
    args = parser.parse_args()

    pair = curve_pair_from_word(parse_artin_word(args.word), base_arc=args.base_arc)
    style = RenderStyle(
        show_base_curve=not args.hide_base,
        moved_opacity=args.moved_opacity,
        moved_linewidth=args.moved_linewidth,
        title=not args.no_title,
    )
    render_curve_pair_with_style(pair, style, out_path=args.out)
    print(f"Word: {pair.word}")
    print(f"Base arc: P_{args.base_arc}")
    print(f"Moved endpoint pair: {pair.moved_endpoint_pair}")
    print(f"Wall-crossing signature: {pair.wall_crossing_signature}")
    print(pair.provenance)
    if args.out is not None:
        print(f"Saved figure to {args.out}")


if __name__ == "__main__":
    main()
