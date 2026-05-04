#!/usr/bin/env python3
"""Local poker equity web app. Run while playing.

Usage:
    python scripts/equity_app.py
    # Then open http://localhost:8765 in your browser

No external deps beyond what's already installed (phevaluator, numpy).
"""

from __future__ import annotations

import json
import sys
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, '.')

from poker.ai.engine.card import Card
from poker.ai.strategy.equity import monte_carlo_equity


PORT = 8765


HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Poker Equity Helper</title>
<style>
  * { box-sizing: border-box; }
  body {
    margin: 0;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: #0d2818;
    color: #e8e8e8;
    padding: 20px;
    min-height: 100vh;
  }
  h1 { margin: 0 0 20px; font-weight: 300; font-size: 24px; }
  .container {
    max-width: 720px;
    margin: 0 auto;
  }
  .panel {
    background: #1a3d2a;
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 20px;
    border: 1px solid #2a5440;
  }
  .panel h2 {
    margin: 0 0 12px;
    font-size: 14px;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: #8fc8a3;
    font-weight: 600;
  }
  .card-slots {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
    margin-bottom: 8px;
  }
  .card-slot {
    width: 50px;
    height: 70px;
    border: 2px dashed #4a7860;
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 22px;
    font-weight: bold;
    cursor: pointer;
    background: #143521;
    transition: all 0.15s;
  }
  .card-slot:hover { border-color: #8fc8a3; }
  .card-slot.filled { border: 2px solid #8fc8a3; background: white; color: black; }
  .card-slot.filled.red { color: #e53935; }
  .card-slot.active { border-color: #ffd54f; box-shadow: 0 0 0 3px rgba(255,213,79,0.3); }
  .deck {
    display: grid;
    grid-template-columns: repeat(13, 1fr);
    gap: 3px;
  }
  .card {
    width: 100%;
    aspect-ratio: 5/7;
    border-radius: 5px;
    background: white;
    color: black;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    cursor: pointer;
    font-size: 13px;
    font-weight: bold;
    transition: transform 0.1s;
    user-select: none;
    border: 1px solid #ccc;
  }
  .card:hover { transform: scale(1.08); }
  .card.red { color: #e53935; }
  .card.used { opacity: 0.2; pointer-events: none; }
  .inputs {
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 12px;
  }
  .inputs label {
    display: flex;
    flex-direction: column;
    font-size: 12px;
    color: #8fc8a3;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }
  .inputs input {
    margin-top: 4px;
    padding: 8px;
    background: #143521;
    border: 1px solid #4a7860;
    border-radius: 6px;
    color: white;
    font-size: 16px;
    font-family: inherit;
  }
  .inputs input:focus { outline: none; border-color: #ffd54f; }
  .result {
    text-align: center;
    padding: 20px;
  }
  .equity {
    font-size: 64px;
    font-weight: 300;
    margin: 0;
    line-height: 1;
  }
  .equity-label { font-size: 12px; color: #8fc8a3; text-transform: uppercase; }
  .recommendation {
    font-size: 24px;
    margin-top: 16px;
    padding: 12px;
    border-radius: 8px;
    font-weight: 600;
  }
  .rec-fold { background: #4a1f1f; color: #ff8a80; }
  .rec-call { background: #1f3d4a; color: #80d8ff; }
  .rec-raise { background: #1f4a2c; color: #69f0ae; }
  .rec-flip { background: #4a3d1f; color: #ffd54f; }
  .meta { font-size: 14px; color: #8fc8a3; margin-top: 8px; }
  .clear-btn {
    background: transparent;
    color: #8fc8a3;
    border: 1px solid #4a7860;
    border-radius: 6px;
    padding: 6px 12px;
    cursor: pointer;
    font-size: 12px;
    margin-left: auto;
  }
  .row { display: flex; align-items: center; gap: 12px; }
  .computing { color: #8fc8a3; font-size: 14px; }
</style>
</head>
<body>
<div class="container">
  <h1>Poker Equity Helper</h1>

  <div class="panel">
    <div class="row">
      <h2 style="margin: 0;">Your hand</h2>
      <button class="clear-btn" onclick="clearHole()">Clear</button>
    </div>
    <div class="card-slots" id="hole-slots">
      <div class="card-slot" data-slot="hole-0"></div>
      <div class="card-slot" data-slot="hole-1"></div>
    </div>
  </div>

  <div class="panel">
    <div class="row">
      <h2 style="margin: 0;">Board</h2>
      <button class="clear-btn" onclick="clearBoard()">Clear</button>
    </div>
    <div class="card-slots" id="board-slots">
      <div class="card-slot" data-slot="board-0"></div>
      <div class="card-slot" data-slot="board-1"></div>
      <div class="card-slot" data-slot="board-2"></div>
      <div class="card-slot" data-slot="board-3"></div>
      <div class="card-slot" data-slot="board-4"></div>
    </div>
  </div>

  <div class="panel">
    <h2>Click any card to assign</h2>
    <div class="deck" id="deck"></div>
  </div>

  <div class="panel">
    <h2>Pot situation</h2>
    <div class="inputs">
      <label>Opponents
        <input type="number" id="opponents" value="1" min="1" max="9">
      </label>
      <label>Pot ($)
        <input type="number" id="pot" value="" placeholder="optional" step="0.01">
      </label>
      <label>Bet to call ($)
        <input type="number" id="to-call" value="" placeholder="optional" step="0.01">
      </label>
    </div>
  </div>

  <div class="panel result" id="result">
    <div class="equity-label">Pick your hand to see equity</div>
  </div>
</div>

<script>
const RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A'];
const SUITS = ['c', 'd', 'h', 's'];
const SUIT_SYMBOLS = { c: '♣', d: '♦', h: '♥', s: '♠' };
const RED_SUITS = new Set(['d', 'h']);

let holeCards = [null, null];
let boardCards = [null, null, null, null, null];
let activeSlot = 'hole-0';
let computeTimeout = null;

function buildDeck() {
  const deck = document.getElementById('deck');
  deck.innerHTML = '';
  for (const rank of RANKS) {
    for (const suit of SUITS) {
      const code = rank + suit;
      const card = document.createElement('div');
      card.className = 'card' + (RED_SUITS.has(suit) ? ' red' : '');
      card.textContent = rank + SUIT_SYMBOLS[suit];
      card.dataset.code = code;
      card.onclick = () => assignCard(code);
      deck.appendChild(card);
    }
  }
}

function assignCard(code) {
  if (cardInUse(code)) return;
  if (activeSlot.startsWith('hole-')) {
    const idx = parseInt(activeSlot.split('-')[1]);
    holeCards[idx] = code;
    activeSlot = idx === 0 ? 'hole-1' : 'board-0';
  } else {
    const idx = parseInt(activeSlot.split('-')[1]);
    boardCards[idx] = code;
    activeSlot = idx < 4 ? `board-${idx + 1}` : null;
  }
  render();
  triggerCompute();
}

function cardInUse(code) {
  return holeCards.includes(code) || boardCards.includes(code);
}

function clearHole() {
  holeCards = [null, null];
  activeSlot = 'hole-0';
  render();
  triggerCompute();
}

function clearBoard() {
  boardCards = [null, null, null, null, null];
  activeSlot = 'board-0';
  render();
  triggerCompute();
}

function clickSlot(slot) {
  activeSlot = slot;
  if (slot.startsWith('hole-')) {
    const idx = parseInt(slot.split('-')[1]);
    holeCards[idx] = null;
  } else {
    const idx = parseInt(slot.split('-')[1]);
    boardCards[idx] = null;
  }
  render();
  triggerCompute();
}

function render() {
  document.querySelectorAll('.card-slot').forEach(s => {
    const slot = s.dataset.slot;
    const isHole = slot.startsWith('hole-');
    const idx = parseInt(slot.split('-')[1]);
    const code = isHole ? holeCards[idx] : boardCards[idx];
    s.classList.remove('filled', 'red', 'active');
    if (code) {
      s.classList.add('filled');
      if (RED_SUITS.has(code[1])) s.classList.add('red');
      s.textContent = code[0] + SUIT_SYMBOLS[code[1]];
    } else {
      s.textContent = '';
    }
    if (slot === activeSlot) s.classList.add('active');
    s.onclick = () => clickSlot(slot);
  });
  document.querySelectorAll('.card').forEach(c => {
    c.classList.toggle('used', cardInUse(c.dataset.code));
  });
}

function triggerCompute() {
  clearTimeout(computeTimeout);
  computeTimeout = setTimeout(compute, 250);
}

async function compute() {
  const hole = holeCards.filter(c => c);
  if (hole.length !== 2) {
    document.getElementById('result').innerHTML =
      '<div class="equity-label">Pick your 2 hole cards</div>';
    return;
  }
  const board = boardCards.filter(c => c);
  if (board.length !== 0 && board.length !== 3 && board.length !== 4 && board.length !== 5) {
    document.getElementById('result').innerHTML =
      '<div class="equity-label">Board needs 0, 3, 4, or 5 cards</div>';
    return;
  }
  const opponents = parseInt(document.getElementById('opponents').value) || 1;
  const pot = parseFloat(document.getElementById('pot').value) || null;
  const toCall = parseFloat(document.getElementById('to-call').value) || null;

  document.getElementById('result').innerHTML =
    '<div class="computing">Computing...</div>';

  const params = new URLSearchParams({
    hole: hole.join(','),
    board: board.join(','),
    opponents: opponents,
  });
  if (pot !== null) params.set('pot', pot);
  if (toCall !== null) params.set('to_call', toCall);

  try {
    const r = await fetch('/equity?' + params);
    const data = await r.json();
    showResult(data);
  } catch (e) {
    document.getElementById('result').innerHTML =
      `<div class="computing" style="color:#ff8a80">Error: ${e.message}</div>`;
  }
}

function showResult(data) {
  let html = `
    <div class="equity-label">Equity vs ${data.opponents} opp</div>
    <div class="equity">${(data.equity * 100).toFixed(1)}%</div>
    <div class="meta">${data.sims} sims · ${data.elapsed_ms}ms · ${data.street}</div>
  `;
  if (data.recommendation) {
    let cls = 'rec-flip';
    if (data.recommendation.startsWith('FOLD')) cls = 'rec-fold';
    else if (data.recommendation.startsWith('CALL')) cls = 'rec-call';
    else if (data.recommendation.startsWith('RAISE') || data.recommendation.startsWith('BET')) cls = 'rec-raise';
    html += `<div class="recommendation ${cls}">${data.recommendation}</div>`;
    if (data.pot_odds !== undefined) {
      html += `<div class="meta">Pot odds: ${(data.pot_odds * 100).toFixed(1)}% · Edge: ${(data.edge * 100).toFixed(0)}%</div>`;
    }
  }
  document.getElementById('result').innerHTML = html;
}

document.getElementById('opponents').addEventListener('input', triggerCompute);
document.getElementById('pot').addEventListener('input', triggerCompute);
document.getElementById('to-call').addEventListener('input', triggerCompute);

buildDeck();
render();
</script>
</body>
</html>
"""


def parse_card_code(code: str) -> int:
    """Convert 'as', 'th', '2c' (lowercase) → int."""
    code = code.strip()
    if len(code) != 2:
        raise ValueError(f"bad card: {code}")
    rank_map = {'2': '2', '3': '3', '4': '4', '5': '5', '6': '6', '7': '7',
                '8': '8', '9': '9', 't': 'T', 'j': 'J', 'q': 'Q', 'k': 'K', 'a': 'A',
                'T': 'T', 'J': 'J', 'Q': 'Q', 'K': 'K', 'A': 'A'}
    return Card.from_str(rank_map[code[0]] + code[1].lower()).to_int()


def compute_equity_response(params: dict) -> dict:
    import time
    hole_strs = params.get('hole', [''])[0].split(',') if params.get('hole') else []
    board_strs = [s for s in (params.get('board', [''])[0].split(',') if params.get('board') else []) if s]
    opponents = int(params.get('opponents', ['1'])[0])
    pot = float(params['pot'][0]) if 'pot' in params and params['pot'][0] else None
    to_call = float(params['to_call'][0]) if 'to_call' in params and params['to_call'][0] else None
    sims = 1000

    hole = [parse_card_code(c) for c in hole_strs if c]
    board = [parse_card_code(c) for c in board_strs if c]

    t = time.time()
    equity = monte_carlo_equity(hole, board, opponents, sims)
    elapsed_ms = int((time.time() - t) * 1000)

    street_name = {0: 'PREFLOP', 3: 'FLOP', 4: 'TURN', 5: 'RIVER'}.get(len(board), 'UNKNOWN')

    response = {
        'equity': equity,
        'opponents': opponents,
        'sims': sims,
        'elapsed_ms': elapsed_ms,
        'street': street_name,
    }

    if pot is not None and to_call is not None and to_call > 0:
        pot_odds = to_call / (pot + to_call)
        edge = equity - pot_odds
        if edge > 0.10:
            rec = f"RAISE — clear favorite (+{int(edge*100)}% edge)"
        elif edge > 0.02:
            rec = f"CALL — profitable (+{int(edge*100)}% edge)"
        elif edge > -0.02:
            rec = "COIN FLIP — call is roughly break-even"
        else:
            rec = f"FOLD — you're behind ({int(edge*100)}% edge)"
        response['recommendation'] = rec
        response['pot_odds'] = pot_odds
        response['edge'] = edge
    elif pot is not None and pot > 0:
        if equity > 0.65:
            response['recommendation'] = f"BET FOR VALUE — bet ${pot*0.67:.2f} (67% pot)"
        elif equity > 0.50:
            response['recommendation'] = f"THIN VALUE — bet ${pot*0.33:.2f} (33% pot)"
        else:
            response['recommendation'] = "CHECK — not strong enough to value bet"

    return response


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        url = urlparse(self.path)
        if url.path == '/':
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.end_headers()
            self.wfile.write(HTML.encode())
        elif url.path == '/equity':
            try:
                params = parse_qs(url.query)
                response = compute_equity_response(params)
                body = json.dumps(response).encode()
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except Exception as e:
                self.send_response(400)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'error': str(e)}).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass  # silence default access log


def main() -> None:
    server = HTTPServer(('localhost', PORT), Handler)
    url = f'http://localhost:{PORT}'
    print(f"Poker Equity Helper running at {url}")
    print(f"Press Ctrl+C to stop.\n")
    try:
        webbrowser.open(url)
    except Exception:
        pass
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
        server.shutdown()


if __name__ == '__main__':
    main()
