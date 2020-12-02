package main

import (
	flag "flag"
	"fmt"
	"github.com/fogleman/imview"
	"github.com/go-gl/gl/v2.1/gl"
	"github.com/go-gl/glfw/v3.1/glfw"
	m "github.com/go-gl/mathgl/mgl64"
	"image"
	imcolor "image/color"
	"math"
	"math/rand"
	os "os"
	"runtime"
	pprof "runtime/pprof"
	"sync"
	"time"
)

const eps = 0.0001

//
// helpers
func RandVec3Sphere() *m.Vec3 {
	u := rand.Float64()*2 - 1
	theta := rand.Float64() * 2 * math.Pi
	return &m.Vec3{
		math.Sqrt(1-u*u) * math.Cos(theta),
		math.Sqrt(1-u*u) * math.Sin(theta),
		u,
	}
}

func RandVec3Hemi(normal *m.Vec3) *m.Vec3 {
	vec := RandVec3Sphere()
	if vec.Dot(*normal) < 0 {
		ivec := vec.Mul(-1)
		return &ivec
	}
	return vec
}

func RandVec3Cone(normal *m.Vec3, angle float64) *m.Vec3 {
	// assume normal is normalized
	rnd := RandVec3Sphere()
	x := rnd.Sub(normal.Mul(normal.Dot(*rnd))).Normalize()
	omega := rand.Float64() * angle
	ret := x.Mul(math.Sin(omega)).Add(normal.Mul(math.Cos(omega)))
	return &ret
}

type Raycastable interface {
	Raycast(pos, ray *m.Vec3) Location
}

type ISDF interface {
	Dist(*m.Vec3) float64
	Norm(*m.Vec3) m.Vec3
}

//
// class Transform
type Transform struct {
	Pos m.Vec3
	Rot m.Quat
}

func (self *Transform) PosToGlobal(vec *m.Vec3) m.Vec3 {
	return qrotate(&self.Rot, vec).Add(self.Pos)
}

func (self *Transform) PosToLocal(vec *m.Vec3) m.Vec3 {
	tvec := vec.Sub(self.Pos)
	return qirotate(&self.Rot, &tvec)
}

func (self *Transform) DirToGlobal(vec *m.Vec3) m.Vec3 {
	return qrotate(&self.Rot, vec)

}

func (self *Transform) DirToLocal(vec *m.Vec3) m.Vec3 {
	return qirotate(&self.Rot, vec)
}

//
// class Object
type Object struct {
	Xform Transform
	Sdf   ISDF
}

//implement Raycastable
func (self *Object) Raycast(pos, ray *m.Vec3) Location {
	mpos := self.Xform.PosToLocal(pos)
	mray := self.Xform.DirToLocal(ray)
	dist := mray.Len()
	dir := mray.Mul(1.0 / dist)
	distMoved := 0.0
	for dist > distMoved {
		usdf := abs(self.Sdf.Dist(&mpos))
		if usdf < eps {
			mnml := self.Sdf.Norm(&mpos)
			return Location{self, self.Xform.PosToGlobal(&mpos), self.Xform.DirToGlobal(&mnml)}
		}
		mpos[0] += dir[0] * usdf
		mpos[1] += dir[1] * usdf
		mpos[2] += dir[2] * usdf
		distMoved += usdf
	}
	return Location{}
}

//
type SceneObject interface {
	Obj() *Object
	Raycastable
}

//
//
type Light struct {
	myObj      *Object
	Brightness float64
	Color      RGBA
}

//implement SceneObject
func (self *Light) Obj() *Object {
	return self.myObj
}

//implement Raycastable
func (self *Light) Raycast(pos, ray *m.Vec3) Location {
	loc := self.myObj.Raycast(pos, ray)
	if loc.Obj != nil {
		loc.Obj = self
	}
	return loc
}

//
type ShadedObject struct {
	myObj  *Object
	Shader BRDF
}

//implement Raycastable
func (self *ShadedObject) Raycast(pos, ray *m.Vec3) Location {
	loc := self.myObj.Raycast(pos, ray)
	if loc.Obj != nil {
		loc.Obj = self
	}
	return loc
}

//implement SceneObject
func (self *ShadedObject) Obj() *Object {
	return self.myObj
}

//
// class BRDF
// surface orientation independant
type BRDF interface {
	Color(in, out, n *m.Vec3) RGBA
	Brdf(in, out, n *m.Vec3) float64
	SampleVec3(in, n *m.Vec3) (ray *m.Vec3, ipdist float64)
}

// lambertian BRDF (uniform energy distribution)
type LambertBRDF struct {
	color RGBA
}

func (self *LambertBRDF) Color(in, out, n *m.Vec3) RGBA {
	return self.color
}

func (self *LambertBRDF) Brdf(in, out, n *m.Vec3) float64 {
	return 1.0 / math.Pi
}

func (self *LambertBRDF) SampleVec3(in, n *m.Vec3) (ray *m.Vec3, ipdist float64) {
	return RandVec3Hemi(n), 2 * math.Pi
}

// Perfect Mirror BRDF
type MirrorBRDF struct {
}

func (self *MirrorBRDF) Color(in, out, n *m.Vec3) RGBA {
	return RGBA{1, 1, 1, 1}
}

func (self *MirrorBRDF) Brdf(in, out, n *m.Vec3) float64 {
	// remember here in is light's output direction and out is light's incoming direction.
	// its like that since we are pathtracing, not scattering photons
	// minus is cuz in looks at surface
	return -1.0 / in.Dot(*n)
}

func (self *MirrorBRDF) SampleVec3(in, n *m.Vec3) (ray *m.Vec3, ipdist float64) {
	ret := in.Sub(n.Mul(2 * n.Dot(*in)))
	return &ret, 1.0
}

// Perfect Refractor BRDF
type RefractBRDF struct {
	ior_inside  float64
	ior_outside float64
	spread      float64
}

func (self *RefractBRDF) Color(in, out, n *m.Vec3) RGBA {
	return RGBA{1, 1, 1, 1}
}
func (self *RefractBRDF) Brdf(in, out, n *m.Vec3) float64 {
	return 1.0 / abs(out.Dot(*n))
}

func (self *RefractBRDF) SampleVec3(in, n *m.Vec3) (ray *m.Vec3, ipdist float64) {
	//ret := in.Sub(n.Mul(2 * n.Dot(*in)))
	//return &ret, 1.0
	//return in, 1.0
	ior_i, ior_o := self.ior_inside, self.ior_outside
	if in.Dot(*n) < 0 {
		ior_i, ior_o = ior_o, ior_i
	}
	ct1 := n.Dot(*in)
	ct12 := ct1 * ct1
	st12 := 1 - ct12
	st22 := st12 * ior_i * ior_i / (ior_o * ior_o)
	if abs(st22) > 1 {
		ret := in.Sub(n.Mul(2 * n.Dot(*in)))
		return &ret, 1.0
	}
	ret := n.Mul(math.Copysign(math.Sqrt(1-st22), ct1)).Add(in.Sub(n.Mul(n.Dot(*in))).Normalize().Mul(math.Sqrt(st22)))
	if self.spread == 0.0 {
		return &ret, 1.0
	} else {
		return RandVec3Cone(&ret, self.spread), 1 //2 * math.Pi * (1 - math.Cos(self.spread))
	}
}

// microfacet
type MicrofacetBRDF struct {
	ag2         float64
	ior_inside  float64
	ior_outside float64
}

func (self *MicrofacetBRDF) Color(in, out, n *m.Vec3) RGBA {
	return RGBA{1, 1, 1, 1}
}

func (self *MicrofacetBRDF) Brdf(in, out, n *m.Vec3) float64 {
	// according to paperL
	// in is -o
	// out is i
	o := in.Mul(-1)

	sameside := out.Dot(*n) * o.Dot(*n)
	ag2m := math.Pow(1.2-0.2*math.Sqrt(abs(o.Dot(*n))), 2) * self.ag2
	if sameside >= 0 { // we are reflecting, either from inside or outside
		hr := o.Add(*out).Normalize().Mul(math.Copysign(1, out.Dot(*n)))
		if o.Dot(*n)*o.Dot(hr) < 0 || out.Dot(*n)*out.Dot(hr) < 0 || n.Dot(hr) < 0 {
			return 0
		}
		//return 1 * (self.fres(&o, &hr, n) * self.beck_g1(out, &hr, n) * self.beck_g1(&o, &hr, n) * self.beck_ddistr(&hr, n)) / (4 * n.Dot(*out) * n.Dot(o))
		return (self.beck_g1(out, &hr, n, ag2m) * self.beck_g1(&o, &hr, n, ag2m)) / (n.Dot(*out) * n.Dot(o))
	} else {
		ior_o, ior_i := self.ior_inside, self.ior_outside
		if out.Dot(*n) < 0 {
			ior_o, ior_i = ior_i, ior_o
		}
		ht := out.Mul(ior_i).Add(o.Mul(ior_o)).Normalize().Mul(-1)

		if o.Dot(*n)*o.Dot(ht) < 0 || out.Dot(*n)*out.Dot(ht) < 0 || n.Dot(ht) < 0 {
			return 0
		}
		//return abs(out.Dot(ht)*o.Dot(ht)/(out.Dot(*n)*o.Dot(*n))) * ior_o * ior_o * (1 - self.fres(&o, &ht, n)) * self.beck_g1(out, &ht, n) * self.beck_g1(&o, &ht, n) * self.beck_ddistr(&ht, n) / t1 / t1
		return abs(o.Dot(ht)/(out.Dot(*n)*o.Dot(*n))) * self.beck_g1(out, &ht, n, ag2m) * self.beck_g1(&o, &ht, n, ag2m)
	}

}

func (self *MicrofacetBRDF) SampleVec3(in, n *m.Vec3) (ray *m.Vec3, ipdist float64) {
	//TODO: make this proper
	//return RandVec3Sphere(), 4 * math.Pi

	//
	i := in.Mul(-1)
	x := *RandVec3Sphere()
	x = x.Sub(n.Mul(n.Dot(x))).Normalize()
	e1 := rand.Float64()
	//// ggx distrib
	//theta := math.Atan2(math.Sqrt(self.ag2*e1), math.Sqrt(1-e1))
	// beck
	ag2m := math.Pow(1.2-0.2*math.Sqrt(abs(i.Dot(*n))), 2) * self.ag2
	theta := math.Atan(math.Sqrt(-ag2m * math.Log(1-e1)))

	mm := n.Mul(math.Cos(theta)).Add(x.Mul(math.Sin(theta)))
	fr := self.fres(&i, &mm, n)
	o := m.Vec3{}

	ior := self.ior_outside / self.ior_inside
	if i.Dot(*n) < 0 {
		ior = 1 / ior
	}
	c := i.Dot(mm)

	forsqrt := 1 + ior*ior*(c*c-1) // same condition in fres, so no point checking here
	if rand.Float64() < fr {       // gen reflected
		o = mm.Mul(2 * i.Dot(mm)).Sub(i)
		//return &o, 4 * abs(o.Dot(mm)/(self.beck_ddistr(&mm, n)*n.Dot(mm))) / fr
		return &o, abs(o.Dot(mm)) / (abs(n.Dot(mm)))
	} else { // gen refracted
		o = mm.Mul(ior*c - math.Copysign(math.Sqrt(forsqrt), i.Dot(*n))).Sub(i.Mul(ior)).Normalize()
		//return &o, 1 / (self.beck_ddistr(&mm, n) * abs(n.Dot(mm))) * math.Pow(i.Dot(mm)+ior*o.Dot(mm), 2) / (ior * ior * abs(o.Dot(mm))) / (1 - fr)
		return &o, 1 / (abs(n.Dot(mm)))
	}
}

func (self *MicrofacetBRDF) fres(i, m, n *m.Vec3) float64 {
	c := abs(i.Dot(*m))
	ior_o, ior_i := self.ior_inside, self.ior_outside
	if i.Dot(*n) < 0 {
		ior_o, ior_i = ior_i, ior_o
	}
	forsqrt := ior_o*ior_o/(ior_i*ior_i) + c*c - 1
	if forsqrt < 0 {
		return 1.0
	}
	//return 0
	g := math.Sqrt(forsqrt)
	cc1 := c*(g+c) - 1
	cc2 := c*(g-c) + 1
	return 0.5 * (g - c) * (g - c) / ((g + c) * (g + c)) * (1 + cc1*cc1/(cc2*cc2))
}

func (self *MicrofacetBRDF) ggx_ddistr(m, n *m.Vec3) float64 {
	mdotn := m.Dot(*n)
	if mdotn <= 0 {
		return 0
	}
	cost2 := mdotn * mdotn
	tant2 := (1 - cost2) / cost2

	return self.ag2 / (math.Pi * cost2 * cost2 * (self.ag2 + tant2) * (self.ag2 + tant2))
}

func (self *MicrofacetBRDF) ggx_g1(v, m, n *m.Vec3) float64 {
	vdotm := v.Dot(*m)
	vdotn := v.Dot(*n)
	if vdotm*vdotn < 0 {
		return 0
	}
	cost2 := vdotn * vdotn
	tant2 := (1 - cost2) / cost2
	return 2.0 / (1 + math.Sqrt(1+self.ag2*tant2))
}

func (self *MicrofacetBRDF) beck_ddistr(m, n *m.Vec3) float64 {
	mdotn := m.Dot(*n)
	if mdotn <= 0 {
		return 0
	}
	cost2 := mdotn * mdotn
	tant2 := (1 - cost2) / cost2

	return math.Exp(-tant2/self.ag2) / (math.Pi * self.ag2 * cost2 * cost2)
}

func (self *MicrofacetBRDF) beck_g1(v, m, n *m.Vec3, ag2m float64) float64 {
	vdotm := v.Dot(*m)
	vdotn := v.Dot(*n)
	if vdotm*vdotn < 0 {
		return 0
	}
	cost2 := vdotn * vdotn
	tant2 := (1 - cost2) / cost2
	a := 1 / math.Sqrt(ag2m*tant2)
	if a < 1.6 {
		return (3.535*a + 2.181*a*a) / (1 + 2.276*a + 2.577*a*a)
	}
	return 1
}

// microfambert
type MicrofambertBRDF struct {
	MicrofacetBRDF
	color RGBA
}

func (self *MicrofambertBRDF) Color(in, out, n *m.Vec3) RGBA {
	return self.color
}

func (self *MicrofambertBRDF) Brdf(in, out, n *m.Vec3) float64 {
	return 1
}

func (self *MicrofambertBRDF) SampleVec3(in, n *m.Vec3) (ray *m.Vec3, ipdist float64) {
	//TODO: make this proper

	//
	i := in.Mul(-1)
	x := *RandVec3Sphere()
	x = x.Sub(n.Mul(n.Dot(x))).Normalize()
	e1 := rand.Float64()
	//// ggx distrib
	//theta := math.Atan2(math.Sqrt(self.ag2*e1), math.Sqrt(1-e1))
	// beck
	ag2m := math.Pow(1.2-0.2*math.Sqrt(abs(i.Dot(*n))), 2) * self.ag2
	theta := math.Atan(math.Sqrt(-ag2m * math.Log(1-e1)))

	mm := n.Mul(math.Cos(theta)).Add(x.Mul(math.Sin(theta)))
	fr := self.fres(&i, &mm, n)
	o := m.Vec3{}

	ior := self.ior_outside / self.ior_inside
	if i.Dot(*n) < 0 {
		ior = 1 / ior
	}
	//c := i.Dot(mm)

	if rand.Float64() < fr { // gen reflected
		o = mm.Mul(2 * i.Dot(mm)).Sub(i)
		ag2m := math.Pow(1.2-0.2*math.Sqrt(abs(i.Dot(*n))), 2) * self.ag2
		return &o, abs(o.Dot(mm)/(n.Dot(mm))) * (self.beck_g1(&o, &mm, n, ag2m) * self.beck_g1(&i, &mm, n, ag2m)) / (n.Dot(o) * n.Dot(i))
	} else { // gen refracted
		return RandVec3Hemi(n), 2 / (1 - fr)
	}
}

//
// class Location
type Location struct {
	Obj      Raycastable
	Pos, Nml m.Vec3
}

//
// Scene class
type Scene struct {
	objects []SceneObject
}

//implement Raycastable
func (scene *Scene) Raycast(pos, ray *m.Vec3) Location {
	loc := Location{}
	for _, obj := range scene.objects {
		newloc := obj.Raycast(pos, ray)
		if newloc.Obj == nil {
			continue
		}
		if loc.Obj == nil || newloc.Pos.Sub(*pos).LenSqr() < loc.Pos.Sub(*pos).LenSqr() {
			loc = newloc
		}
	}
	return loc
}

func (scene *Scene) AddObject(obj SceneObject) {
	scene.objects = append(scene.objects, obj)
}

// Sphere implementation
type Sphere struct {
	radius float64
}

func (self *Sphere) Dist(pos *m.Vec3) float64 {
	return pos.Len() - self.radius
}

func (self *Sphere) Norm(pos *m.Vec3) m.Vec3 {
	return pos.Normalize()
}

// Box object class
type Box struct {
	Sx, Sy, Sz float64
}

func (self *Box) locvec(pos m.Vec3) m.Vec3 {
	// sign := m.Vec3{-1 + 2*int(pos[0] > 0), -1 + 2*int(pos[1] > 0), -1 + 2*int(pos[2] > 0)}
	pos[0] = math.Copysign(max(0, math.Abs(pos[0])-self.Sx), pos[0])
	pos[1] = math.Copysign(max(0, math.Abs(pos[1])-self.Sy), pos[1])
	pos[2] = math.Copysign(max(0, math.Abs(pos[2])-self.Sz), pos[2])
	return pos
}

func (self *Box) distVec(pos *m.Vec3) m.Vec3 {
	return m.Vec3{math.Abs(pos[0]) - self.Sx/2, math.Abs(pos[1]) - self.Sy/2, math.Abs(pos[2]) - self.Sz/2}
}

func (self *Box) Dist(pos *m.Vec3) float64 {
	//distvec := self.distVec(pos)
	//return max(distvec[0], max(distvec[1], distvec[2]))
	return max(math.Abs(pos[0])-self.Sx/2,
		max(
			math.Abs(pos[1])-self.Sy/2,
			math.Abs(pos[2])-self.Sz/2,
		),
	)
	//return max(math.Abs(pos[0])-self.Sx/2, max(math.Abs(pos[1])-self.Sy/2, math.Abs(pos[2])-self.Sz/2))
}

func (self *Box) Norm(pos *m.Vec3) m.Vec3 {
	dv := self.distVec(pos)
	switch {
	case dv[0] >= dv[1] && dv[0] >= dv[2]:
		return m.Vec3{math.Copysign(1, pos[0]), 0, 0}
	case dv[1] >= dv[0] && dv[1] >= dv[2]:
		return m.Vec3{0, math.Copysign(1, pos[1]), 0}
	default:
		return m.Vec3{0, 0, math.Copysign(1, pos[2])}
	}
}

//
// convenience functions
func MakeObject(x, y, z, rx, ry, rz float64, shape ISDF) *Object {
	obj := Object{Transform{m.Vec3{x, y, z}, m.AnglesToQuat(rx, ry, rz, m.XYZ)}, shape}
	return &obj
}

func MakeShadedObject(brdf BRDF, obj *Object) *ShadedObject {
	sobj := ShadedObject{myObj: obj, Shader: brdf}
	return &sobj
}

func MakeLight(intens float64, color RGBA, obj *Object) *Light {
	sobj := Light{myObj: obj, Brightness: intens, Color: color}
	return &sobj
}

type RGBA struct {
	R, G, B, A float64
}

func (clr RGBA) Add(clr2 RGBA) RGBA {
	return RGBA{clr.R + clr2.R,
		clr.G + clr2.G,
		clr.B + clr2.B,
		clr.A + clr2.A,
	}
}

func (clr *RGBA) AddToThis(clr2 RGBA) *RGBA {
	clr.R += clr2.R
	clr.G += clr2.G
	clr.B += clr2.B
	clr.A += clr2.A
	return clr
}

func (clr RGBA) RGBmul(factor float64) RGBA {
	clr.R *= factor
	clr.G *= factor
	clr.B *= factor
	return clr
}

func (clr RGBA) RGBAmulRGBA(clr2 RGBA) RGBA {
	return RGBA{clr.R * clr2.R,
		clr.G * clr2.G,
		clr.B * clr2.B,
		clr.A * clr2.A,
	}
}

func (clr RGBA) RGBAmul(factor float64) RGBA {
	clr.R *= factor
	clr.G *= factor
	clr.B *= factor
	clr.A *= factor
	return clr
}

func (clr RGBA) ToAnotherRGBA() imcolor.RGBA {
	return imcolor.RGBA{uint8(math.Min(255, (clr.R * 255))), uint8(math.Min(255, (clr.G * 255))), uint8(math.Min(255, (clr.B * 255))), uint8(math.Min(255, (clr.A * 255)))}
}

type SampleInfo struct {
	clr  RGBA
	x, y int
}

//
// main guy
const mrl = 10

func pathtrace(scene *Scene, pos, ray *m.Vec3, depth uint) RGBA {
	bgcolor := RGBA{0, 0, 0, 0}
	longray := ray.Mul(mrl)
	loc := scene.Raycast(pos, &longray)
	if loc.Obj == nil {
		return bgcolor
	}
	l, ok := loc.Obj.(*Light)
	if ok {
		return l.Color.RGBmul(l.Brightness)
	}
	if depth == 30 {
		return RGBA{0, 0, 0, 0}
	}
	shadedobj, ok := loc.Obj.(*ShadedObject)
	if ok {
		newray, iprob := shadedobj.Shader.SampleVec3(ray, &loc.Nml)
		newpos := loc.Pos.Add(loc.Nml.Mul(2 * math.Copysign(eps, newray.Dot(loc.Nml))))
		light := pathtrace(scene, &newpos, newray, depth+1)
		if abs(loc.Nml.Len()-1.0) > 1e-3 || abs(newray.Len()-1.0) > 1e-3 {
			fmt.Println("panik!")
		}
		costheta := abs(loc.Nml.Dot(*newray))

		//probability := 1.0 / (2.0 * math.Pi) //this is prob distribution
		return light.RGBmul(shadedobj.Shader.Brdf(ray, newray, &loc.Nml) * iprob * costheta).RGBAmulRGBA(shadedobj.Shader.Color(ray, newray, &loc.Nml))
	}
	fmt.Println("how?")
	return RGBA{1, 0, 0, 1}
}

func gatherer(clrout chan SampleInfo, x, y int, scene *Scene, pos, baseray m.Vec3, wigglex m.Vec3, wiggley m.Vec3, nsamples int, wg *sync.WaitGroup) {
	//avgcolor := RGBA{}
	for i := 0; i < nsamples; i++ {
		ray := baseray.Add(wigglex.Mul(rand.Float64() - 0.5)).Add(wiggley.Mul(rand.Float64() - 0.5)).Normalize()
		//avgcolor = avgcolor.Add(pathtrace(scene, &pos, &ray, 0))
		clrout <- SampleInfo{pathtrace(scene, &pos, &ray, 0), x, y}
	}
	//img.SetRGBA(x, y, avgcolor.RGBAmul(1.0/float64(nsamples)).ToAnotherRGBA())
	wg.Done()
}

func gatherDequeuer(tasks chan func()) {
	for fu := range tasks {
		fu()
	}
	fmt.Println("dequeuer done, fucking off")
}

func imageWriter(img *image.RGBA, width, height int, ch chan SampleInfo) {
	accum := make([]RGBA, width*height)
	count := make([]uint, width*height)
	for s := range ch {
		off := s.y*width + s.x
		count[off]++
		accum[off].AddToThis(s.clr)
		img.SetRGBA(s.x, s.y, accum[off].RGBAmul(1.0/float64(count[off])).ToAnotherRGBA())
		//if accum[off].G/float64(count[off]) > 5 {
		//	fmt.Println(s.x, s.y, accum[off], count[off])
		//}
	}
	fmt.Println("imageWriter out")
}

var cpuprofile = flag.String("cpuprofile", "", "write cpu profile to file")
var memprofile = flag.String("memprofile", "", "write mem profile to file")
var nsamplesarg = flag.Int("n", 1, "sample count")

func main() {
	flag.Parse()
	if *cpuprofile != "" {
		f, err := os.Create(*cpuprofile)
		if err != nil {
			fmt.Println(err)
			return
		}
		pprof.StartCPUProfile(f)
		defer pprof.StopCPUProfile()
	}

	// gl init for picture display
	if err := gl.Init(); err != nil {
		return
	}

	if err := glfw.Init(); err != nil {
		return
	}
	defer glfw.Terminate()
	fmt.Println("gl init success")
	// end gl init

	fmt.Println("vim-go")
	imgWidth := 1024
	imgHeight := 1024
	var img *image.RGBA = image.NewRGBA(image.Rect(0, 0, imgWidth, imgHeight))
	v := m.Vec3{1, 2, 3}
	v = v.Add(m.Vec3{2, 3, 4})
	fmt.Println(v)

	scene := Scene{}
	scene.AddObject(MakeShadedObject(&MicrofacetBRDF{ag2: math.Pow(0.1, 2), ior_inside: 1.51, ior_outside: 1.0}, &Object{Transform{m.Vec3{0.09, 0.175, -0.1}, m.AnglesToQuat(0, 0, 0, m.XYZ)}, &Sphere{0.1}}))
	//scene.AddObject(MakeShadedObject(&RefractBRDF{ior_inside: 1.51, ior_outside: 1.0, spread: 0.0}, &Object{Transform{m.Vec3{0.1, 0.2, 0}, m.AnglesToQuat(0, 0, 0, m.XYZ)}, &Sphere{0.1}}))
	scene.AddObject(MakeShadedObject(&MicrofacetBRDF{ag2: math.Pow(0.001, 2), ior_inside: 1.51, ior_outside: 1.0}, &Object{Transform{m.Vec3{0.05, -0.2, -0.1}, m.AnglesToQuat(0, 0, 0, m.XYZ)}, &Sphere{0.05}}))
	scene.AddObject(MakeShadedObject(&MicrofambertBRDF{MicrofacetBRDF{ag2: math.Pow(0.05, 2), ior_inside: 1.85, ior_outside: 1.0}, RGBA{0.95, 0.95, 0.95, 1}}, &Object{Transform{m.Vec3{-0.05, -0.25, -0.1}, m.AnglesToQuat(0, 0, 0, m.XYZ)}, &Sphere{0.05}}))
	//scene.AddObject(MakeShadedObject(&LambertBRDF{}, &Object{Transform{m.Vec3{-0.05, -0.25, -0.1}, m.AnglesToQuat(0, 0, 0, m.XYZ)}, &Sphere{0.05}}))
	//&Object{Transform{m.Vec3{0.0, 0, 0}, m.AnglesToQuat(0.7, -0.2, 0.7, m.XYZ)}
	scene.AddObject(MakeShadedObject(&MirrorBRDF{}, MakeObject(0, -0.15, 0, 0.7, -0.2, 0.7, &Box{0.1, 0.1, 0.1})))
	scene.AddObject(MakeShadedObject(&LambertBRDF{RGBA{0.8, 0.8, 0.8, 1}}, MakeObject(0.0, 0.305, 0.5, 0, 0, 0, &Box{1.0, 0.01, 1.5})))
	scene.AddObject(MakeShadedObject(&LambertBRDF{RGBA{0.8, 0.8, 0.8, 1}}, MakeObject(0.0, -0.305, 0.5, 0, 0, 0, &Box{1.0, 0.01, 1.5})))
	scene.AddObject(MakeShadedObject(&LambertBRDF{RGBA{0.8, 0.8, 0.8, 1}}, MakeObject(0.0, 0.0, 0.2, 0, 0, 0, &Box{1.0, 1.0, 0.01})))
	scene.AddObject(MakeShadedObject(&LambertBRDF{RGBA{0.8, 0.8, 0.8, 1}}, MakeObject(-0.1, -0.3, 0.2, 0, 0, 0, &Box{0.025, 0.4, 0.5})))
	scene.AddObject(MakeShadedObject(&LambertBRDF{RGBA{0.8, 0.8, 0.8, 1}}, MakeObject(0.1, -0.3, 0.2, 0, 0, 0, &Box{0.025, 0.4, 0.5})))

	//scene.AddObject(MakeShadedObject(&LambertBRDF{}, MakeObject(-0.2, 0, 0, 0, 0, 0, &Box{0.01, 0.6, 0.4})))
	scene.AddObject(MakeShadedObject(&LambertBRDF{RGBA{0.8, 0.8, 0.8, 1}}, MakeObject(0.2, 0, 0, 0, 0, 0, &Box{0.01, 0.6, 0.4})))
	scene.AddObject(MakeLight(1.0, RGBA{1.0, 0.1, 0.1, 1.0}, MakeObject(-0.2, 0, 0, 0, 0, 0, &Box{0.01, 0.6, 0.4})))
	//scene.AddObject(MakeLight(1.0, RGBA{0.1, 1.0, 0.1, 1.0}, MakeObject(0.2, 0, 0, 0, 0, 0, &Box{0.01, 0.6, 0.4})))

	scene.AddObject(MakeLight(10.0, RGBA{0.3, 1.0, 0.2, 1.0}, MakeObject(0, 0.0, 0, 0, 0, 0, &Sphere{0.05})))
	//scene.AddObject(&Light{*MakeObject(0.0, 0.3, 0.5, 0, 0, 0, &Box{1.0, 0.01, 1.5}), 1.0, RGBA{0.1, 0.1, 1.0, 1.0}})

	//behind the camera white light
	//scene.AddObject(MakeLight(1.0, RGBA{1.0, 1.0, 1.0, 1.0}, MakeObject(0, 0, -0.9, 0, 0, 0, &Box{20, 20, 0.2})))

	fmt.Println(scene)

	camPos := m.Vec3{0, 0, -0.5}
	camRot := m.AnglesToQuat(0, 0, 0, m.XYZ)
	camFocal := 0.5

	nsamples := *nsamplesarg
	startTime := time.Now()
	onecallsamples := 4
	if nsamples < onecallsamples {
		onecallsamples = 1
	}
	shiftOff := 41
	shiftRounds := nsamples / onecallsamples
	waiter := sync.WaitGroup{}
	waiter.Add(1) // i so group is not empty, we'll remove it after everything is go-ed

	taskqueue := make(chan func(), 1000)
	for i := 0; i < 32; i++ {
		go gatherDequeuer(taskqueue)
	}
	samplequeue := make(chan SampleInfo, 128)
	go imageWriter(img, imgWidth, imgHeight, samplequeue)
	go finisher(startTime, &waiter, samplequeue)

	go func() {
		for shift := 0; shift < shiftOff*shiftRounds; shift++ {
			for idx := shift % shiftOff; idx < imgWidth*imgHeight; idx += shiftOff {
				x := idx % imgWidth
				y := idx / imgWidth
				camFwd := camRot.Rotate(m.Vec3{0, 0, 1})
				screenVec := camRot.Rotate(m.Vec3{float64(x)/float64(imgWidth) - 0.5, -float64(y)/float64(imgHeight) + 0.5, 0}).Add(camFwd.Mul(camFocal))

				waiter.Add(1)
				taskqueue <- func(clrout chan SampleInfo, x, y int, scene *Scene, pos, baseray m.Vec3, wigglex m.Vec3, wiggley m.Vec3, nsamples int, wg *sync.WaitGroup) func() {
					return func() { gatherer(clrout, x, y, scene, pos, baseray, wigglex, wiggley, nsamples, wg) }
				}(
					samplequeue,
					x,
					y,
					&scene,
					camPos,
					screenVec,
					camRot.Rotate(m.Vec3{1.0 / float64(imgWidth)}),
					camRot.Rotate(m.Vec3{0, 1.0 / float64(imgHeight)}),
					onecallsamples,
					&waiter)
			}

		}
		close(taskqueue)
		waiter.Done() // remove that 1 we added before loop
		fmt.Println("scheduling done. waiting for the waiter", time.Since(startTime))
	}()

	//display
	window, err := imview.NewWindow(img)
	if err != nil {
		return
	}

	for {
		if window.ShouldClose() {
			break
		}
		window.SetImage(img)
		window.Draw()
		glfw.PollEvents()
	}

	// mem profile
	if *memprofile != "" {
		f, err := os.Create(*memprofile)
		if err != nil {
			fmt.Println(err)
			return
		}
		defer f.Close()
		runtime.GC()
		if err := pprof.WriteHeapProfile(f); err != nil {
			fmt.Println(err)
			return
		}
	}
}

func finisher(startTime time.Time, waiter *sync.WaitGroup, samplequeuetoclose chan SampleInfo) {
	waiter.Wait()
	close(samplequeuetoclose)
	fmt.Println("total time:", time.Since(startTime))
}

// bullshit
func max(a, b float64) float64 {
	if a > b {
		return a
	}
	return b
}

func abs(a float64) float64 {
	if a < 0 {
		return -a
	}
	return a
}

func qrotate(q *m.Quat, v *m.Vec3) m.Vec3 {
	// need to write it as simple as possible
	// simplify the formula
	//replacement for q.Rotate(*v)
	tq := m.Quat{W: -q.V[0]*v[0] - q.V[1]*v[1] - q.V[2]*v[2],
		V: m.Vec3{q.W*v[0] + q.V[1]*v[2] - q.V[2]*v[1],
			q.W*v[1] - q.V[0]*v[2] + q.V[2]*v[0],
			q.W*v[2] + q.V[0]*v[1] - q.V[1]*v[0],
		}}
	res := m.Vec3{
		tq.W*q.V[0] + tq.V[0]*(-q.W) + tq.V[1]*q.V[2] - tq.V[2]*q.V[1],
		tq.W*q.V[1] - tq.V[0]*q.V[2] + tq.V[1]*(-q.W) + tq.V[2]*q.V[0],
		tq.W*q.V[2] + tq.V[0]*q.V[1] - tq.V[1]*q.V[0] + tq.V[2]*(-q.W),
	}
	return res
}

func qirotate(q *m.Quat, v *m.Vec3) m.Vec3 {
	// need to write it as simple as possible
	// simplify the formula
	// replacement for q.Inverse().Rotate(*v)
	tq := m.Quat{W: -q.V[0]*v[0] - q.V[1]*v[1] - q.V[2]*v[2],
		V: m.Vec3{(-q.W)*v[0] + q.V[1]*v[2] - q.V[2]*v[1],
			(-q.W)*v[1] - q.V[0]*v[2] + q.V[2]*v[0],
			(-q.W)*v[2] + q.V[0]*v[1] - q.V[1]*v[0],
		}}
	res := m.Vec3{
		tq.W*q.V[0] + tq.V[0]*(+q.W) + tq.V[1]*q.V[2] - tq.V[2]*q.V[1],
		tq.W*q.V[1] - tq.V[0]*q.V[2] + tq.V[1]*(+q.W) + tq.V[2]*q.V[0],
		tq.W*q.V[2] + tq.V[0]*q.V[1] - tq.V[1]*q.V[0] + tq.V[2]*(+q.W),
	}
	return res
}
