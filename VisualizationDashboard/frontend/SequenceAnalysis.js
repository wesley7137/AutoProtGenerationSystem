import React, { useState, useEffect } from "react";
import { useParams } from "react-router-dom";
import { Paper, Typography, Grid, CircularProgress } from "@material-ui/core";
import axios from "axios";
import NglViewer from "./NglViewer";

function SequenceAnalysis() {
  const { id } = useParams();
  const [sequence, setSequence] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchSequence = async () => {
      setLoading(true);
      try {
        const result = await axios.get(`/api/sequences/${id}`);
        setSequence(result.data);
      } catch (error) {
        console.error("Error fetching sequence:", error);
      }
      setLoading(false);
    };

    fetchSequence();
  }, [id]);

  if (loading) return <CircularProgress />;
  if (!sequence) return <Typography>Sequence not found</Typography>;

  return (
    <Grid container spacing={3}>
      <Grid item xs={12}>
        <Paper>
          <Typography variant="h4">Sequence Analysis: {sequence.id}</Typography>
          <Typography>Sequence: {sequence.sequence}</Typography>
          <Typography>Score: {sequence.score.toFixed(2)}</Typography>
          <Typography>Status: {sequence.status}</Typography>
        </Paper>
      </Grid>
      <Grid item xs={12}>
        <Paper>
          <Typography variant="h5">Structure Visualization</Typography>
          <NglViewer pdbUrl={sequence.pdbUrl} />
        </Paper>
      </Grid>
    </Grid>
  );
}

export default SequenceAnalysis;
