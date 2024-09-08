import React from "react";
import { BrowserRouter as Router, Route, Switch } from "react-router-dom";
import { ThemeProvider, createMuiTheme } from "@material-ui/core/styles";
import CssBaseline from "@material-ui/core/CssBaseline";
import Dashboard from "./components/Dashboard";
import SequenceAnalysis from "./components/SequenceAnalysis";
import Header from "./components/Header";
import Notifications from "./components/Notifications";

const theme = createMuiTheme({
  palette: {
    type: "dark",
    primary: {
      main: "#424242",
    },
    secondary: {
      main: "#4caf50",
    },
    background: {
      default: "#303030",
      paper: "#424242",
    },
  },
});

function App() {
  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Router>
        <Header />
        <Notifications />
        <Switch>
          <Route exact path="/" component={Dashboard} />
          <Route path="/sequence/:id" component={SequenceAnalysis} />
        </Switch>
      </Router>
    </ThemeProvider>
  );
}

export default App;
