import jsPDF from 'jspdf';
import html2canvas from 'html2canvas';

/**
 * Generate a PDF report for a player by capturing a hidden div.
 * The div is temporarily added to the DOM, captured, then removed.
 */
export async function generatePlayerPDF(playerData) {
  const { row, score, band, bandColor, offScore, midScore, defScore, marketValue, scoutingText, season } = playerData;

  // Create off-screen container
  const container = document.createElement('div');
  container.style.cssText = `
    position: fixed;
    top: -9999px;
    left: -9999px;
    width: 794px;
    background: white;
    font-family: system-ui, -apple-system, sans-serif;
    color: #111;
    padding: 0;
  `;

  container.innerHTML = `
    <div style="background: #0b1f1e; color: white; padding: 16px 24px; display: flex; justify-content: space-between; align-items: center;">
      <div style="font-size: 18px; font-weight: 700; letter-spacing: -0.02em;">
        <span style="color: #00B8A9;">Player</span>Score
      </div>
      <div style="font-size: 11px; color: #80F5E3;">Player Report · ${new Date().toLocaleDateString('en-GB', { day: 'numeric', month: 'short', year: 'numeric' })}</div>
    </div>

    <div style="padding: 24px;">
      <div style="font-size: 26px; font-weight: 800; margin-bottom: 4px;">${row.Player}</div>
      <div style="font-size: 13px; color: #555; margin-bottom: 20px; display: flex; gap: 12px;">
        <span>${row.Squad}</span><span>·</span><span>${row.Comp}</span>
        <span>·</span><span>${row.Pos}</span><span>·</span>
        <span>Age ${Math.round(parseFloat(row.Age))}</span><span>·</span><span>${season}</span>
      </div>

      <div style="display: flex; gap: 24px; margin-bottom: 24px;">
        <div style="flex: 0 0 200px; background: #f4f9f9; border-radius: 12px; padding: 20px; text-align: center;">
          <div style="font-size: 52px; font-weight: 800; color: #00B8A9; line-height: 1;">${Math.round(score)}</div>
          <div style="font-size: 11px; color: #555; margin-top: 4px;">PlayerScore · Range 0–1000</div>
          <div style="margin-top: 12px; font-size: 13px; font-weight: 700; color: ${bandColor}; background: ${bandColor}22; border: 1px solid ${bandColor}44; border-radius: 999px; padding: 4px 12px; display: inline-block;">${band}</div>
          ${marketValue ? `<div style="margin-top: 10px; font-size: 12px; color: #555;">Market Value: <strong>${marketValue >= 1_000_000 ? '€' + (marketValue / 1_000_000).toFixed(0) + 'M' : '€' + (marketValue / 1_000).toFixed(0) + 'K'}</strong></div>` : ''}
        </div>

        <div style="flex: 1;">
          <div style="font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em; color: #555; margin-bottom: 12px;">Score Breakdown</div>
          ${[['Offense', offScore], ['Midfield', midScore], ['Defense', defScore]].map(([label, val]) => `
            <div style="margin-bottom: 12px;">
              <div style="display: flex; justify-content: space-between; font-size: 12px; margin-bottom: 4px;">
                <span style="font-weight: 600;">${label}</span>
                <span style="color: #555;">${Math.round(val)}</span>
              </div>
              <div style="background: #e5e7eb; border-radius: 4px; height: 8px; overflow: hidden;">
                <div style="background: #00B8A9; width: ${Math.min(val / 10, 100)}%; height: 100%; border-radius: 4px;"></div>
              </div>
            </div>
          `).join('')}

          <div style="margin-top: 16px; font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em; color: #555; margin-bottom: 8px;">Playing Time</div>
          <div style="font-size: 14px; font-weight: 600;">${parseFloat(row['90s'] || 0).toFixed(1)} <span style="font-size: 11px; color: #555; font-weight: 400;">× 90 minutes played</span></div>
        </div>
      </div>

      <div style="background: #edf7f6; border-left: 3px solid #00B8A9; border-radius: 0 8px 8px 0; padding: 16px; margin-bottom: 16px;">
        <div style="font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em; color: #00897B; margin-bottom: 8px;">Scouting Summary</div>
        <div style="font-size: 13px; line-height: 1.6; color: #222;">${scoutingText}</div>
      </div>

      <div style="font-size: 10px; color: #999; border-top: 1px solid #e5e7eb; padding-top: 12px; display: flex; justify-content: space-between;">
        <span>Data: FBref · Big-5 European Leagues · Transfermarkt</span>
        <span>PlayerScore — twinanalytics.github.io/player-score</span>
      </div>
    </div>
  `;

  document.body.appendChild(container);

  try {
    const canvas = await html2canvas(container, {
      scale: 2,
      useCORS: true,
      backgroundColor: '#ffffff',
    });

    const imgData = canvas.toDataURL('image/png');
    const pdf = new jsPDF({ orientation: 'portrait', unit: 'mm', format: 'a4' });

    const pageWidth = pdf.internal.pageSize.getWidth();
    const pageHeight = pdf.internal.pageSize.getHeight();
    const imgWidth = pageWidth;
    const imgHeight = (canvas.height * pageWidth) / canvas.width;

    pdf.addImage(imgData, 'PNG', 0, 0, imgWidth, Math.min(imgHeight, pageHeight));

    const filename = `PlayerScore_${row.Player.replace(/\s+/g, '_')}_${season}.pdf`;
    pdf.save(filename);
  } finally {
    document.body.removeChild(container);
  }
}
