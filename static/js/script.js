 // Create a request variable and assign a new XMLHttpRequest object to it.
 //ID to access the interval .Now set as 5 sec for each request
var intervalId = window.setInterval(getAllData, 5000);

var getAllData = function(){
    var request = new XMLHttpRequest()

    // Open a new connection, using the GET request on the URL endpoint
    request.open('GET', 'http://localhost:5000/getAllData', true);

    request.onload = function () {  // Event Listener for loading
      var data = JSON.parse(this.response);
      console.log(data);
      const gendercontainer = document.getElementById('genderresults')//Result output
      gendercontainer.innerHTML = 'Gender: ';
      data.Gender.forEach((gender) => {
          const span = document.createElement('span');
          span.textContent = gender+"  ";
          gendercontainer.appendChild(span)
      });

      const agecontainer = document.getElementById('ageresults')//Result output
      agecontainer.innerHTML = 'Age: ';
      data.Age.forEach((age) => {
          const span = document.createElement('span');
          span.textContent = age+"  ";
          agecontainer.appendChild(span)
      });

      const emotioncontainer = document.getElementById('emotionresults')//Result output
      emotioncontainer.innerHTML = 'Emotion: ';
      data.Emotion.forEach((emotion) => {
          const span = document.createElement('span');
          span.textContent = emotion+"  ";
          emotioncontainer.appendChild(span)
      });
    }
    // Send request
  request.send()
}

