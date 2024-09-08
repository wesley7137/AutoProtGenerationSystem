import React, { useState, useEffect } from "react";
import {
  Grid,
  Paper,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Typography,
  TextField,
  Button,
  CircularProgress,
} from "@material-ui/core";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
} from "recharts";
import axios from "axios";
import { Link } from "react-router-dom";

function Dashboard() {
  const [sequences, setSequences] = useState([]);
  const [scoreData, setScoreData] = useState([]);
  const [loading, setLoading] = useState(true);
  const [filter, setFilter] = useState("");

  useEffect(() => {
    const fetchData = async () => {
      setLoading(true);
      try {
        const result = await axios.get("/api/sequences");
        setSequences(result.data);
        const scores = result.data.map((seq, index) => ({
          name: index,
          score: seq.score,
        }));
        setScoreData(scores);
      } catch (error) {
        console.error("Error fetching data:", error);
      }
      setLoading(false);
    };

    fetchData();
    const interval = setInterval(fetchData, 5000); // Refresh every 5 seconds

    return () => clearInterval(interval);
  }, []);

  const filteredSequences = sequences.filter(
    (seq) =>
      seq.sequence.includes(filter) ||
      seq.status.toLowerCase().includes(filter.toLowerCase()) ||
      seq.score.toString().includes(filter)
  );

  return (
    <Grid container spacing={3}>
      <Grid item xs={12}>
        <Paper>
          <Typography variant="h5">Sequence Scores Over Time</Typography>
          <LineChart width={600} height={300} data={scoreData}>
            <XAxis dataKey="name" />
            <YAxis />
            <CartesianGrid strokeDasharray="3 3" />
            <Tooltip />
            <Legend />
            <Line type="monotone" dataKey="score" stroke="#8884d8" />
          </LineChart>
        </Paper>
      </Grid>
      <Grid item xs={12}>
        <Paper>
          <TextField
            label="Filter sequences"
            variant="outlined"
            fullWidth
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
          />
          <TableContainer>
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell>ID</TableCell>
                  <TableCell>Sequence</TableCell>
                  <TableCell>Score</TableCell>
                  <TableCell>Status</TableCell>
                  <TableCell>Actions</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {loading ? (
                  <TableRow>
                    <TableCell colSpan={5} align="center">
                      <CircularProgress />
                    </TableCell>
                  </TableRow>
                ) : (
                  filteredSequences.map((seq) => (
                    <TableRow key={seq.id}>
                      <TableCell>{seq.id}</TableCell>
                      <TableCell>{seq.sequence}</TableCell>
                      <TableCell>{seq.score.toFixed(2)}</TableCell>
                      <TableCell>{seq.status}</TableCell>
                      <TableCell>
                        <Button
                          component={Link}
                          to={`/sequence/${seq.id}`}
                          variant="contained"
                          color="primary"
                        >
                          Analyze
                        </Button>
                      </TableCell>
                    </TableRow>
                  ))
                )}
              </TableBody>
            </Table>
          </TableContainer>
        </Paper>
      </Grid>
    </Grid>
  );
}

export default Dashboard;
