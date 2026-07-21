type Row = {
  key: string;
  cells: (string | number)[];
};

export default function ResultsTable({
  caption,
  columns,
  rows,
}: {
  caption?: string;
  columns: string[];
  rows: Row[];
}) {
  return (
    <div className="table-scroll">
      <table className="data-table">
        {caption && <caption>{caption}</caption>}
        <thead>
          <tr>
            {columns.map((col) => (
              <th key={col}>{col}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={row.key}>
              {row.cells.map((cell, i) => (
                <td key={i}>{cell}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
