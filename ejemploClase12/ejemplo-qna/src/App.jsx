import React, { useState, useEffect, useRef } from 'react'
import './App.css'

import * as tf from '@tensorflow/tfjs'
import * as qna from '@tensorflow-models/qna'
import {Puff} from 'react-loader-spinner'
import { Fragment } from 'react'

function App() {
  const passageRef = useRef(null)
  const questionRef = useRef(null)
  const [answer, setAnswer] = useState(null)
  const [model, setModel] = useState(null)

  const loadModel = async() => {
    const loadedModel = await qna.load()
    setModel(loadedModel)
    console.log('Modelo cargado')
  }

  useEffect(()=>{loadModel()}, [])

  const answerQuestion = async (e) =>{
    if(e.which === 13 && model !== null){
      console.log("Pregunta enviada")
      const passage = passageRef.current.value
      const question = questionRef.current.value

      const answers = await model.findAnswers(question, passage)
      setAnswer(answers)
      console.log(answers)
    }
  }

  return (
    <div>
      {
        model == null ?
        <div>
          <div>Cargando modelo...</div>
          <Puff
            color='#00BFFF'
            height={100}
            width={100}
          />
        </div>
        :
        <Fragment>
          <textarea ref={passageRef} rows="30" cols="100"></textarea>
          <input ref={questionRef} onKeyDown={answerQuestion} size="80"></input>
          {answer ? answer.map((ans, idx)=><div><b>Respuesta {idx+1} - </b> {ans.text} {ans.score}</div>):""}
        </Fragment>
      }
    </div>
  )
}

export default App
