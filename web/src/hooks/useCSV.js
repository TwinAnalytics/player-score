import { useState, useEffect } from 'react';
import Papa from 'papaparse';

export function useCSV(url) {
  const [state, setState] = useState({ data: null, loading: true, error: null });

  useEffect(() => {
    if (!url) return;
    setState({ data: null, loading: true, error: null });

    Papa.parse(url, {
      download: true,
      header: true,
      skipEmptyLines: true,
      dynamicTyping: false,
      complete: (results) => {
        setState({ data: results.data, loading: false, error: null });
      },
      error: (err) => {
        setState({ data: null, loading: false, error: err.message || 'Failed to load data' });
      },
    });
  }, [url]);

  return state;
}
