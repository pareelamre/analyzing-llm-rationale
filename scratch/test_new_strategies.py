import json
import sys
from pathlib import Path

def load_data():
    with open("static/track_record_live.json") as f:
        return json.load(f)

sys.path.insert(0, str(Path("src").resolve()))
from analyzing_llm_rationale import track_record_live as trl

def main():
    data = load_data()
    bets = data["paper_pnl"]["bets"]
    bets_sorted = sorted(bets, key=lambda x: x.get("resolved_ts") or "")
    
    def run_sim(sizing_type, filter_fn=None):
        staked = 0.0
        pnl = 0.0
        wins = 0
        n = 0
        history = []
        
        for b in bets_sorted:
            raw_model_p = b["model_probability"]
            mkt_p = b["market_probability"]
            side = b["side"]
            win_val = b["win"]
            
            # 1. Calibrated model probability
            if len(history) >= 30:
                pairs = [(float(x["model_probability"]), int(x["outcome"])) for x in history]
                bp = trl._fit_isotonic(pairs)
                model_p = trl._apply_isotonic(bp, raw_model_p)
            else:
                model_p = raw_model_p
                
            # Calibrated edge vs raw edge
            cal_edge = abs(model_p - mkt_p)
            raw_edge = b["edge"]
            
            # Apply filters if any
            if filter_fn and not filter_fn(b):
                history.append(b)
                continue
                
            if sizing_type == "raw_edge_weighted":
                stake = min(raw_edge, 0.25)
            elif sizing_type == "calibrated_edge_weighted":
                stake = min(cal_edge, 0.25)
            else:
                stake = 1.0
                
            if stake <= 0.0:
                history.append(b)
                continue
                
            p_side = mkt_p if side == "YES" else (1.0 - mkt_p)
            if p_side <= 0.0 or p_side >= 1.0:
                history.append(b)
                continue
                
            fee = trl._bet_fee(b.get("platform"), stake, p_side)
            odds = (1.0 - p_side) / p_side
            profit = stake * (odds if win_val else -1.0) - fee
            
            staked += stake
            pnl += profit
            if win_val:
                wins += 1
            n += 1
            history.append(b)
            
        roi = pnl / staked if staked > 0 else 0.0
        wr = wins / n if n > 0 else 0.0
        return {"n_bets": n, "total_staked": staked, "pnl": pnl, "roi": roi, "win_rate": wr}

    def _smart_filter(b):
        mkt_p = b["market_probability"]
        p_side = mkt_p if b["side"] == "YES" else (1.0 - mkt_p)
        if p_side < 0.20 or p_side > 0.80:
            return False
        if b.get("domain") == "geopolitics" and b["edge"] > 0.10:
            return False
        if b["edge"] > 0.40:
            return False
        return True

    print("=== EDGE-WEIGHTED UPGRADE SIMULATIONS ===")
    print()
    
    # 1. Current Raw Edge Weighted
    raw_ew = run_sim("raw_edge_weighted")
    print(f"Raw Edge-Weighted:  n={raw_ew['n_bets']:3d}, PnL={raw_ew['pnl']:+8.2f}, ROI={raw_ew['roi']*100:>+6.1f}%, Staked=${raw_ew['total_staked']:.2f}")
    
    # 2. Calibrated Edge Weighted (No Filters)
    cal_ew = run_sim("calibrated_edge_weighted")
    print(f"Calib Edge-Weighted: n={cal_ew['n_bets']:3d}, PnL={cal_ew['pnl']:+8.2f}, ROI={cal_ew['roi']*100:>+6.1f}%, Staked=${cal_ew['total_staked']:.2f}")

    # 3. Calibrated Edge Weighted + Smart Filters
    cal_ew_smart = run_sim("calibrated_edge_weighted", filter_fn=_smart_filter)
    print(f"Smart Edge-Weighted: n={cal_ew_smart['n_bets']:3d}, PnL={cal_ew_smart['pnl']:+8.2f}, ROI={cal_ew_smart['roi']*100:>+6.1f}%, Staked=${cal_ew_smart['total_staked']:.2f}")

if __name__ == "__main__":
    main()
